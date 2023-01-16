import os
import numpy as np
import torch
import argparse
import pkg_resources

from typing import Union, List
from operator import itemgetter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from synapse_augmenter import SynapseAugmenter
from synapse_dataset import SynapseDataset

from synapse_simclr.modules import SynapseSimCLR, NT_Xent, generate_resnet_3d
import synapse_simclr.utils as utils

# how to log?
log_info = print


class SynapseSimCLRWorkspace:
    
    def __init__(
            self,
            gpu_index: int,
            args: argparse.Namespace):

        assert args.run_mode in {'pretrain', 'extract'}
        
        log_info(f"Instantiating the Synapse SimCLR workspace in {args.run_mode} mode ...")

        if args.run_mode == 'extract':
            shuffle = False
            drop_last = False
            train_mode = False
            
        elif args.run_mode == 'pretrain':
            shuffle = True
            drop_last = True
            train_mode = True
            
        else:
            raise ValueError

        # rank of the process
        rank = args.nr * args.gpus + gpu_index

        if args.nodes > 1:
            dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
            torch.cuda.set_device(gpu_index)

        # make a dictionary of augmenter options after argparse
        augmenter_config_keys = utils.yaml_config_hook(args.augmenter_config_yaml_path).keys()
        args_dict = vars(args)
        augmenter_kwargs = {key: args_dict[key] for key in augmenter_config_keys}
    
        # set the random seed for determinism
        if augmenter_kwargs['batch_mode'] == 'coupled':
            
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            
        else:
            
            torch.manual_seed(args.seed + args.nr)
            np.random.seed(args.seed + args.nr)

        # load the dataset
        log_info("Instantiating synapse dataset ...")
        train_dataset = SynapseDataset(
            dataset_path=args.dataset_path,
            head=args.dataset_head if args.dataset_head > 0 else None,
            drop_annotated_synapses=(args.drop_annotated_synapses > 0))
        log_info(f'- Training dataset size: {len(train_dataset)}')
            
        # instantiate data sampler and loader
        if args.nodes > 1:
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=args.world_size,
                rank=rank,
                shuffle=shuffle)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=drop_last,
                collate_fn=lambda x: x,  # do not collate; return as a list of np.ndarray
                num_workers=args.dataloader_workers,
                sampler=train_sampler,
                pin_memory=True)

        else:
            
            train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=lambda x: x,  # do not collate; return as a list of np.ndarray
                num_workers=args.dataloader_workers,
                sampler=None,
                pin_memory=True)

        # instantiate the augmenter
        augmenter = SynapseAugmenter(
            **augmenter_kwargs,
            device=args.device,
            dtype=args.dtype)

        # instantiate the encoder
        log_info("Instantiating the encoder ...")
        encoder = utils.instantiate_encoder(
            encoder_type=args.encoder_type,
            n_input_channels=augmenter.n_intensity_output_channels)

        # instantiate the SynapseSimCLR model
        log_info("Instantiating SimCLR ...")
        model = SynapseSimCLR(encoder, args.projection_dims).to(args.device).type(args.dtype)
        model.train(mode=train_mode)

        # instantiate the optimizer and LR scheduler
        log_info("Instantiating the optimizer and scheduler ...")
        optimizer, scheduler = utils.instantiate_optimizer(
            optimizer_type=args.optimizer_type,
            model=model,
            adam_lr=args.adam_lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            epochs=args.epochs)
        
        # make sure checkpoint path exists
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path, exist_ok=True)

        # reload from a checkpoint?
        if int(args.reload):
            
            # load model
            model_checkpoint_path = os.path.join(
                args.checkpoint_path,
                f"model_checkpoint_{args.reload_epoch_num}.pt")        
            log_info(f"Loading model checkpoint from {model_checkpoint_path} ...")
            model_state_dict = torch.load(model_checkpoint_path, map_location=args.device.type)
            # ..note:: we need to drop "module." prefix from model state dict keys
            #   for some reason, saving the module attribute of DDP models adds this extra prefix ...
            fixed_model_state_dict = dict()
            for key, value in model_state_dict.items():
                fixed_model_state_dict[key[7:] if key.find("module.") == 0 else key] = value
            
            model_keys = set(map(itemgetter(0), model.named_parameters()))
            loaded_model_keys = set(fixed_model_state_dict.keys())
            missing_keys = model_keys.difference(loaded_model_keys)
            if len(missing_keys) == 0:
                log_info("All model parameters are available in the loaded state dictionary.")
            else:
                log_info("WARNING: The following model parameters are not available in the loaded state dictionary:")
                for key in missing_keys:
                    log_info(f'* {key}')
            model.load_state_dict(fixed_model_state_dict, strict=False)

            # load optimizer state?
            if int(args.reload_optimizer_state):
                
                optimizer_checkpoint_path = os.path.join(
                    args.checkpoint_path,
                    f"optimizer_checkpoint_{args.reload_epoch_num}.pt")
                log_info(f"Loading optimizer checkpoint from {optimizer_checkpoint_path} ...")
                optimizer_state_dict = torch.load(optimizer_checkpoint_path, map_location=args.device.type)
                optimizer.load_state_dict(optimizer_state_dict)

                scheduler_checkpoint_path = os.path.join(
                    args.checkpoint_path,
                    f"scheduler_checkpoint_{args.reload_epoch_num}.pt")
                log_info(f"Loading scheduler checkpoint from {scheduler_checkpoint_path} ...")
                scheduler_checkpoint_path = torch.load(scheduler_checkpoint_path, map_location=args.device.type)
                scheduler.load_state_dict(scheduler_checkpoint_path)

                if args.optimizer_type == "Adam":
                    lr = scheduler.get_last_lr()[0]
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                else:
                    raise NotImplementedError("Only reloading Adam optimizer state is supported!")
            
            else:
                log_info("Optimizer state NOT reloaded.")
            
            
            self.start_epoch = args.reload_epoch_num + 1

        else:
            
            assert args.run_mode == 'pretrain', \
                'If not pretraining, you must reload a pretrained model from a checkpoint!'
            
            self.start_epoch = 0

        # loss
        criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

        # DDP
        if args.nodes > 1:
            
            # only sync batch norm if need to train; otherwise, no need
            if train_mode:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                
            model = DDP(model, device_ids=[gpu_index])
            model = model.to(args.device)
            model.train(mode=train_mode)

        if args.nr == 0:
            # todo: placeholder for instantiating a summary writer
            pass
        
        if args.nodes > 1:
            module = model.module
        else:
            module = model
        
        if args.nr == 0:
            log_info("\n")
            log_info("Encoder architecture:\n")
            log_info(module.encoder)
            log_info("\n")
            log_info("Projector architecture:\n")
            log_info(module.projector)
            log_info("\n")

        # references to useful variables
        self.nr = args.nr
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.augmenter = augmenter
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

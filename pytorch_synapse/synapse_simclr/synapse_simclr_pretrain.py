import os
import numpy as np
import torch
import argparse
import pkg_resources
from operator import itemgetter

from typing import Union, List

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from synapse_simclr import SynapseSimCLRWorkspace
import synapse_simclr.utils as utils

torch.backends.cudnn.benchmark = True

# if in notebook mode, use default arguments
notebook_mode = False

# how to log?
log_info = print
        

def run_synapse_simclr_pretrain(
        gpu_index: int,
        args: argparse.Namespace):
    """Stages and runs the Synapse SimCLR pretraining loop."""
    
    # instantiate the workspace
    ws = SynapseSimCLRWorkspace(gpu_index, args)
    
    log_info(f"Hashes before starting pretraining (make sure different nodes have the same hash) ...\n")    
    utils.print_hash(
        model=ws.model,
        optimizer=ws.optimizer,
        scheduler=ws.scheduler)
    log_info(f"\n")    

    # the training loop
    log_info(f"Starting the training loop ...")
    for i_epoch in range(ws.start_epoch, args.epochs):
        
        if ws.train_sampler is not None:
            ws.train_sampler.set_epoch(i_epoch)
            
        ws.model.train()
            
        lr = ws.optimizer.param_groups[0]["lr"]

        log_info(f"Start epoch [{i_epoch}/{args.epochs}]\t lr: {lr:.7f}")
        
        loss_list = run_synapse_simclr_pretrain_single_epoch(i_epoch, args, ws)
        loss_epoch = np.sum(loss_list)
        
        if ws.scheduler is not None:
            ws.scheduler.step()
            
        if i_epoch % args.checkpoint_frequency == 0:

            utils.print_hash(
                model=ws.model,
                optimizer=ws.optimizer,
                scheduler=ws.scheduler)

            if args.nr == 0:
                
                log_info(f"Checkpointing ...")
                utils.checkpoint_state(
                    model=ws.model,
                    optimizer=ws.optimizer,
                    scheduler=ws.scheduler,
                    output_path=args.checkpoint_path,
                    current_epoch=i_epoch)
                
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if i_epoch % args.summary_frequency == 0:

            if args.nr == 0:
                
                log_info(f"Writing summary ...")
                utils.write_summary(
                    output_path=args.checkpoint_path,
                    loss_list=loss_list,
                    current_epoch=i_epoch)

        log_info(
            f"End epoch [{i_epoch}/{args.epochs}]\t "
            f"Loss: {loss_epoch / len(ws.train_loader)}\t lr: {lr:.7f}")

    # end training
    if args.nr == 0:
        
        log_info(f"Checkpointing ...")
        utils.checkpoint_state(
            model=ws.model,
            optimizer=ws.optimizer,
            scheduler=ws.scheduler,
            output_path=args.checkpoint_path,
            current_epoch=i_epoch)

    utils.print_hash(
        model=ws.model,
        optimizer=ws.optimizer,
        scheduler=ws.scheduler)
    
    log_info(f"Training finished!")

    
def run_synapse_simclr_pretrain_single_epoch(
        i_epoch: int,
        args: argparse.Namespace,
        ws: SynapseSimCLRWorkspace) -> List[float]:
    """Runs a single epoch of Synapse SimCLR pretraining and returns the minibatch loss list."""

    loss_list = []
    for i_minibatch, minibatch_data_bundle in enumerate(ws.train_loader):
        
        # obtain two independent augmentations from the loaded data
        # .. note:: minibatch_data_bundle is a list of Tuple[int, np.ndarray]
        #     (i.e. index and data) we drop the index
        # .. note:: the second return argument is the mask and is ignored
        
        if args.print_minibatch_indices:
            index_list = list(map(itemgetter(0), minibatch_data_bundle))
            info_string = f"Node rank: {args.nr}, epoch index: {i_epoch}, minibatch index: {i_minibatch}"
            separator = "=" * len(info_string)
            log_info(info_string)
            log_info(separator)
            log_info(', '.join(list(map(str, index_list))))
            log_info(separator)
            
        intensity_mask_cxyz_list = list(map(itemgetter(1), minibatch_data_bundle))
        x_i, _ = ws.augmenter.augment_raw_data(intensity_mask_cxyz_list)
        x_j, _ = ws.augmenter.augment_raw_data(intensity_mask_cxyz_list)
        
        # SimCLR step
        ws.optimizer.zero_grad(set_to_none=True)
        
        z_list_i = ws.model(x_i)
        z_list_j = ws.model(x_j)
        
        z_projector_head_i = z_list_i[-1]
        z_projector_head_j = z_list_j[-1]
        
        loss = ws.criterion(z_projector_head_i, z_projector_head_j)
        
        loss.backward()
        ws.optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and i_minibatch % args.loss_logging_frequency == 0:
            log_info(
                f"Minibatch [{i_minibatch}/{len(ws.train_loader)}]\t"
                f"Loss: {loss.item()}")
        
        loss_list.append(loss.item())
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return loss_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Synapse SimCLR")
    
    parser.add_argument(f"--config-yaml-path", type=str, required=True)
    
    # for DDP
    parser.add_argument(f"--master-addr", default="127.0.0.1", type=str)
    parser.add_argument(f"--master-port", default="8000", type=str)
    parser.add_argument(f"--nodes", default=1, type=int)
    parser.add_argument(f"--gpus", default=1, type=int)    
    parser.add_argument(f"--nr", default=0, required=True, type=int)

    primary_args = parser.parse_args(args=[] if notebook_mode else None)
    
    simclr_config_yaml_path = os.path.join(
        primary_args.config_yaml_path, 'config.yaml')
    augmenter_config_yaml_path = os.path.join(
        primary_args.config_yaml_path, 'augmenter_pretrain.yaml')
    
    log_info(f"Loading base SimCLR configuration from {simclr_config_yaml_path} ...")
    log_info(f"Loading base SynapseAugmenter configuration from {augmenter_config_yaml_path} ...")

    simclr_config = utils.yaml_config_hook(simclr_config_yaml_path)
    augmenter_config = utils.yaml_config_hook(augmenter_config_yaml_path)

    for key, value in simclr_config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))

    for key, value in augmenter_config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))

    args = parser.parse_args(args=[] if notebook_mode else None)
    
    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    # inject additional keys into the args Namespace
    args.augmenter_config_yaml_path = augmenter_config_yaml_path
    args.nr = primary_args.nr
    args.run_mode = 'pretrain'
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.dtype = utils.parse_dtype(args.dtype)
    
    if args.nodes > 1:
        
        log_info(
            f"Training with {args.nodes} nodes, waiting until all nodes "
            f"join before starting training ...")
        mp.spawn(
            run_synapse_simclr_pretrain,
            args=(args,),
            nprocs=args.gpus,
            join=True)
    
    else:
        
        run_synapse_simclr_pretrain(0, args)

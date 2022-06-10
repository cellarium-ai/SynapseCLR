import os
import numpy as np
import torch
import argparse
import pkg_resources
from operator import itemgetter
from typing import Union, List, Dict, Iterable, Callable
from collections import defaultdict

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from synapse_simclr import SynapseSimCLRWorkspace
import synapse_simclr.utils as utils


# if in notebook mode, use default arguments
notebook_mode = False

# how to log?
log_info = print


class FeatureExtractor(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Module,
            args: argparse.Namespace,
            layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            if args.world_size > 1:
                fetch_layer_id = "module." + layer_id
            else:
                fetch_layer_id = layer_id
            layer = dict([*self.model.named_modules()])[fetch_layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features
    
    
class FeatureBuffer:
    
    def __init__(self):
        self._buffer_dict = defaultdict(list)
    
    def accumulate(
            self,
            index_tensor: torch.Tensor,
            feature_dict: Dict[str, torch.Tensor]):
        self._buffer_dict['index_array'].append(index_tensor.cpu().numpy())
        for key, value in feature_dict.items():
            self._buffer_dict[key].append(value.cpu().numpy())

    def _validate(self):
        full_index_array_n = np.concatenate(self._buffer_dict['index_array'])
        assert full_index_array_n.ndim == 1
        n_samples = full_index_array_n.shape[0]
        for key in self._buffer_dict.keys():
            n_samples_for_key = 0
            for value in self._buffer_dict[key]:
                n_samples_for_key += value.shape[0]
            assert n_samples_for_key == n_samples, \
                f'Key {key} has {n_samples_for_key} buffered values; expected {n_samples}'

    def save(
            self,
            output_path: str,
            suffix: str,
            validate: bool = True):
        if validate:
            self._validate()
        for key, values in self._buffer_dict.items():
            concat_values = np.concatenate(values, axis=0)
            output_file_path = os.path.join(
                output_path,
                f'extracted_features__{suffix}__{key}.npy')
            np.save(output_file_path, concat_values)


def get_feature_dict_list_mean(feature_dict_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = feature_dict_list[0].keys()
    mean_feature_dict = dict()
    for key in keys:
        mean_feature_value = torch.mean(
            torch.cat(
                [feature_dict[key][None, :] for feature_dict in feature_dict_list],
                dim=0),
            dim=0)
        mean_feature_dict[key] = mean_feature_value
    return mean_feature_dict
    
    
def run_synapse_simclr_extract_features(
        gpu_index: int,
        args: argparse.Namespace):
    """Stages and runs the Synapse SimCLR feature extraction loop."""
    
    # instantiate the workspace
    ws = SynapseSimCLRWorkspace(gpu_index, args)
    
    # assert that output path exists
    if not os.path.exists(args.feature_output_path):
        os.makedirs(args.feature_output_path, exist_ok=True)
    
    # forward feture extraction wrapper
    module = ws.model.module if isinstance(ws.model, torch.torch.nn.DataParallel) else ws.model
    feature_extractor = FeatureExtractor(
        model=module,
        args=args,
        layers=args.feature_extraction_hooks)
    
    # a buffer for extracted features
    feature_buffer = FeatureBuffer()
    
    # the training loop
    log_info(f"Starting the feature extraction loop ...")
    
    if ws.train_sampler is not None:
        ws.train_sampler.set_epoch(0)
    
    ws.model.eval()
    
    for i_minibatch, minibatch_data_bundle in enumerate(ws.train_loader):
        
        # obtain two independent augmentations from the loaded data
        # .. note:: minibatch_data_bundle is a list of Tuple[int, np.ndarray]
        #     (i.e. index and data) we drop the index
        # .. note:: the second return argument is the mask and is ignored
        
        index_tensor = torch.tensor(list(map(itemgetter(0), minibatch_data_bundle)))

        if args.print_minibatch_indices:
            index_list = index_tensor.numpy().tolist()
            info_string = f"Node rank: {args.nr}, minibatch index: {i_minibatch}"
            separator = "=" * len(info_string)
            log_info(info_string)
            log_info(separator)
            log_info(', '.join(list(map(str, index_list))))
            log_info(separator)
            
        # extract features
        intensity_mask_cxyz_list = list(map(itemgetter(1), minibatch_data_bundle))
        feature_dict_list = []
        with torch.no_grad():
            for _ in range(args.n_extract_augmentations):
                x, _ = ws.augmenter.augment_raw_data(intensity_mask_cxyz_list)
                feature_dict_list.append(feature_extractor(x))
        feature_dict = get_feature_dict_list_mean(feature_dict_list)
        
        # send to buffer
        feature_buffer.accumulate(
            index_tensor=index_tensor,
            feature_dict=feature_dict)

        if args.nr == 0 and i_minibatch % args.extract_logging_frequency == 0:
            log_info(f"Minibatch [{i_minibatch}/{len(ws.train_loader)}]")
        
    # save buffer
    log_info("Saving features ...")
    feature_buffer.save(
        output_path=args.feature_output_path,
        suffix=f'node.{args.nr}__epoch.{args.reload_epoch_num}')
    
    log_info(f"Feature extraction finished!")
    
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
        primary_args.config_yaml_path, 'augmenter_extract.yaml')
    
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
    args.run_mode = 'extract'
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.dtype = utils.parse_dtype(args.dtype)
    
    if args.nodes > 1:
        
        log_info(
            f"Training with {args.nodes} nodes, waiting until all nodes "
            f"join before starting training ...")
        mp.spawn(
            run_synapse_simclr_extract_features,
            args=(args,),
            nprocs=args.gpus,
            join=True)
    
    else:
        
        run_synapse_simclr_extract_features(0, args)

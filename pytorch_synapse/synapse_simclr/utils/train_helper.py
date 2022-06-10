from typing import List, Tuple, Union, Optional

import os
import hashlib
import pickle
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from synapse_simclr.modules import \
    LARS, \
    generate_resnet_3d

_SUPPORTED_ENCODERS = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet18_no_max_pool',
    'resnet34_no_max_pool',
    'resnet50_no_max_pool',
    'resnet18_no_max_pool_cifar_stem',
    'resnet34_no_max_pool_cifar_stem',
    'resnet50_no_max_pool_cifar_stem',
    'resnet18_medicalnet'
]


def instantiate_encoder(
        encoder_type: str,
        n_input_channels: int,
        **kwargs) -> torch.nn.Module:
    
    if encoder_type == 'resnet18':
        
        return generate_resnet_3d(
            model_depth=18,
            n_input_channels=n_input_channels,
            no_max_pool=False,
            conv1_kernel_size=7,
            conv1_stride=2,
            conv1_padding=3,
            **kwargs)
    
    if encoder_type == 'resnet34':
        
        return generate_resnet_3d(
            model_depth=34,
            n_input_channels=n_input_channels,
            no_max_pool=False,
            conv1_kernel_size=7,
            conv1_stride=2,
            conv1_padding=3,
            **kwargs)

    elif encoder_type == 'resnet50':
        
        return generate_resnet_3d(
            model_depth=50,
            n_input_channels=n_input_channels,
            no_max_pool=False,
            conv1_kernel_size=7,
            conv1_stride=2,
            conv1_padding=3,
            **kwargs)
    
    elif encoder_type == 'resnet18_no_max_pool':
        
        return generate_resnet_3d(
            model_depth=18,
            n_input_channels=n_input_channels,
            no_max_pool=True,
            conv1_kernel_size=7,
            conv1_stride=2,
            conv1_padding=3,
            **kwargs)

    elif encoder_type == 'resnet34_no_max_pool':
        
        return generate_resnet_3d(
            model_depth=34,
            n_input_channels=n_input_channels,
            no_max_pool=True,
            conv1_kernel_size=7,
            conv1_stride=2,
            conv1_padding=3,
            **kwargs)

    elif encoder_type == 'resnet50_no_max_pool':
        
        return generate_resnet_3d(
            model_depth=50,
            n_input_channels=n_input_channels,
            no_max_pool=True,
            conv1_kernel_size=7,
            conv1_stride=2,
            conv1_padding=3,
            **kwargs)

    elif encoder_type == 'resnet18_no_max_pool_cifar_stem':
        
        return generate_resnet_3d(
            model_depth=18,
            n_input_channels=n_input_channels,
            no_max_pool=True,
            conv1_kernel_size=3,
            conv1_stride=1,
            conv1_padding=1,
            **kwargs)

    elif encoder_type == 'resnet34_no_max_pool_cifar_stem':
        
        return generate_resnet_3d(
            model_depth=34,
            n_input_channels=n_input_channels,
            no_max_pool=True,
            conv1_kernel_size=3,
            conv1_stride=1,
            conv1_padding=1,
            **kwargs)
    
    elif encoder_type == 'resnet50_no_max_pool_cifar_stem':
        
        return generate_resnet_3d(
            model_depth=50,
            n_input_channels=n_input_channels,
            no_max_pool=True,
            conv1_kernel_size=3,
            conv1_stride=1,
            conv1_padding=1,
            **kwargs)
    
    elif encoder_type == 'resnet18_medicalnet':
        
        return generate_resnet_3d(
            model_depth=18,
            n_input_channels=n_input_channels,
            no_max_pool=False,
            conv1_kernel_size=7,
            conv1_stride=2,
            conv1_padding=3,
            shortcut_type='A',
            block_dilations=[1, 1, 2, 4],
            block_strides=[1, 2, 1, 1])
    
    else:
        
        raise ValueError(
            f'Only the following encoders are currently supported: '
            f'{", ".join(_SUPPORTED_ENCODERS)}')


def instantiate_optimizer(
        optimizer_type: str,
        model: Union[torch.nn.Module, torch.nn.DataParallel],
        adam_lr: Optional[float] = 3e-4,
        batch_size: Optional[int] = None,
        weight_decay: Optional[float] = None,
        epochs: Optional[int] = None) -> Tuple[Optimizer, Optional[CosineAnnealingLR]]:

    assert optimizer_type in {"Adam", "LARS"}
    
    if optimizer_type == "Adam":
        assert adam_lr is not None

        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
        
        # decay the learning rate with the cosine decay schedule without restarts
        eta_min = 0.1 * adam_lr
        scheduler = CosineAnnealingLR(
            optimizer, epochs, eta_min=eta_min, last_epoch=-1)

    elif optimizer_type == "LARS":
        assert batch_size is not None
        assert weight_decay is not None
        assert epochs is not None
        
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"])
        
        # decay the learning rate with the cosine decay schedule without restarts
        eta_min = 0.
        scheduler = CosineAnnealingLR(
            optimizer, epochs, eta_min=eta_min, last_epoch=-1)
        
    else:
        raise ValueError


    return optimizer, scheduler


def checkpoint_state(
        model: Union[torch.nn.Module, torch.nn.DataParallel],
        optimizer: Optimizer,
        scheduler: CosineAnnealingLR,
        output_path: str,
        current_epoch: int):
    
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    module = model.module if isinstance(model, torch.torch.nn.DataParallel) else model
    
    torch.save(
        module.state_dict(),
        os.path.join(output_path, f"model_checkpoint_{current_epoch}.pt"))
    
    torch.save(
        optimizer.state_dict(),
        os.path.join(output_path, f"optimizer_checkpoint_{current_epoch}.pt"))
    
    torch.save(
        scheduler.state_dict(),
        os.path.join(output_path, f"scheduler_checkpoint_{current_epoch}.pt"))


def get_cpu_state_dict(state_dict: dict) -> dict:
    cpu_state_dict = dict()
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu_state_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            cpu_state_dict[k] = get_cpu_state_dict(v)
        else:
            cpu_state_dict[k] = v
    return cpu_state_dict
    
def print_hash(
        model: Union[torch.nn.Module, torch.nn.DataParallel],
        optimizer: Optimizer,
        scheduler: CosineAnnealingLR):
    
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    module = model.module if isinstance(model, torch.torch.nn.DataParallel) else model

    model_md5_hash = hashlib.md5(pickle.dumps(get_cpu_state_dict(module.state_dict()))).hexdigest()
    optimizer_md5_hash = hashlib.md5(pickle.dumps(get_cpu_state_dict(optimizer.state_dict()))).hexdigest()
    scheduler_md5_hash = hashlib.md5(pickle.dumps(get_cpu_state_dict(scheduler.state_dict()))).hexdigest()

    print(f'model MD5 hash: {model_md5_hash}')
    print(f'optimizer MD5 hash: {optimizer_md5_hash}')
    print(f'scheduler MD5 hash: {scheduler_md5_hash}')


def write_summary(
        output_path: str,
        loss_list: List[float],
        current_epoch: int):
    np.save(
        os.path.join(output_path, f"loss_list_{current_epoch}.npy"),
        np.asarray(loss_list))

    
def parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == 'float16':
        return torch.float16
    elif dtype_str == 'float32':
        return torch.float32
    elif dtype_str == 'float64':
        return torch.float64
    else:
        raise ValueError

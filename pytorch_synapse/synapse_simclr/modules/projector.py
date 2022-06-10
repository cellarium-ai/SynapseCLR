from typing import List
import torch


class Projector(torch.nn.Module):
    
    def __init__(
            self,
            input_features: int,
            channel_dims: List[int],
            bias: bool = False):
        """MLP projection head for SimCLR."""
        
        super(Projector, self).__init__()
        
        # generate a basic MLP
        blocks = []
        prev_channel = input_features
        for channel_dim in channel_dims[:-1]:
            blocks.append(torch.nn.Linear(prev_channel, channel_dim, bias=bias))
            blocks.append(torch.nn.BatchNorm1d(channel_dim))
            blocks.append(torch.nn.ReLU())
            prev_channel = channel_dim
        blocks.append(torch.nn.Linear(prev_channel, channel_dims[-1], bias=bias))
        self.mlp = torch.nn.Sequential(*blocks)
        
    def forward(self, x) -> List[torch.Tensor]:
        """
        .. note: returns a list of activations from all linear layers; the last entry
          on the list is the one that is used in SimCLR.  
        """
        layers = list(self.mlp.children())
        linear_activations = []
        for layer in layers:
            x = layer(x)
            if isinstance(layer, torch.nn.Linear):
                linear_activations.append(x)
        return linear_activations
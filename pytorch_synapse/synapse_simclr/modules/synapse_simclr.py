import os
from scipy.stats import truncnorm
import numpy as np
from typing import List, Tuple, Union, Optional
import torch
from synapse_simclr.modules import Identity, Projector


class SynapseSimCLR(torch.nn.Module):
    """TBW.
    """

    def __init__(
            self,
            encoder: torch.nn.Module,
            projection_dims: List[int]):
        """
        
        .. note::
          encoder must have the following attributes:
            - 'fc': a fully connected readout layer (that we strip out)
            - 'n_features': number of features just after the final pooling and before 'fc'
                  
        """
        super(SynapseSimCLR, self).__init__()

        self.encoder = encoder
        self.projection_dims = projection_dims

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with ReLU non-linearity as the projection head
        self.projector = Projector(
            input_features=self.encoder.n_features,
            channel_dims=projection_dims)
        
        self.reset_parameters()

    def reset_parameters(self):
        def conv3d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1. / fan_in) / .87962566103423978
            r = truncnorm.rvs(-2, 2, loc=0, scale=1., size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                p.normal_(std=0.01)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                conv3d_weight_truncated_normal_init(m.weight)
            elif isinstance(m, torch.nn.Linear):
                linear_normal_init(m.weight)
    
    @property
    def output_dims(self) -> List[int]:
        encoder_output_dim = self.encoder.n_features
        projector_output_dims = self.projection_dims
        return [encoder_output_dim] + projector_output_dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        .. note: returns a list of activations; the first entry on the list is the output from
          the backbone (e.g. ResNet), and the last entry is the output from the last layer of the
          project.
        """
        # generate representations for the two sets of augmentations
        h = self.encoder(x)

        # project to the "cosine similarity space"
        z_list = self.projector(h)
        
        return [h] + z_list

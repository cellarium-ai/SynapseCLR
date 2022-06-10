import os
from scipy.stats import truncnorm
import numpy as np
from typing import List, Tuple, Union, Optional
import torch
from synapse_simclr.modules import SynapseSimCLR


class SynapseSupervised(torch.nn.Module):
    """TBW.
    """

    def __init__(
            self,
            synapse_simclr: SynapseSimCLR,
            synapse_simclr_output_layer_index: int,
            linear_readout_type: List[str],
            linear_readout_num_categories: List[int]):
        """
        
        :param synapse_simclr_output_layer_index: which layer of the SimCLR model to construct
          linear predictors from?
        :param linear_readout_type: 'continous' or 'categorical'
        :param linear_readout_num_categories: for 'continuous', must be 1; for categorical, however many
          categories there are
        """
        super(SynapseSupervised, self).__init__()

        self.synapse_simclr = synapse_simclr
        
        
    
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

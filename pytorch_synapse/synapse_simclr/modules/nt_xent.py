import torch
import torch.distributed as dist
from .gather import GatherLayer


class NT_Xent(torch.nn.Module):
    
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.full_batch_size = world_size * batch_size
        
        self.mask = self.get_self_excluding_mask()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def get_self_excluding_mask(self) -> torch.Tensor:
        mask = torch.ones((2 * self.full_batch_size, 2 * self.full_batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.full_batch_size):
            mask[i, i + self.full_batch_size] = 0
            mask[i + self.full_batch_size, i] = 0
        return mask

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:

        if self.world_size > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        
        # first M entries of z will be the first augmentation; last M entries will
        # be the second augmentation
        z = torch.cat((z_i, z_j), dim=0)
        
        # the dimension of sim is (2 * self.full_batch_size) x (2 * self.full_batch_size)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.full_batch_size)
        sim_j_i = torch.diag(sim, -self.full_batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * self.full_batch_size, 1)
        negative_samples = sim[self.mask].reshape(2 * self.full_batch_size, -1)

        labels = torch.zeros(2 * self.full_batch_size).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels) / (2 * self.full_batch_size)
        
        return loss

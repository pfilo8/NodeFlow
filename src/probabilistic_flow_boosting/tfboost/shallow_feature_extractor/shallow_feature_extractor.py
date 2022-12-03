import torch
import torch.nn as nn


class ShallowFeatureExtractor(nn.Module):

    def __init__(self, input_dim, output_dim, device):
        super().__init__()

        self.device = device

        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
        )
        self.model = self.model.to(self.device)

    def forward(self, x):
        return self.model.forward(x)

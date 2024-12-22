# domain_encoder.py
import torch.nn as nn

class DomainEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DomainEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
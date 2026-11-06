"""
model.py
========
Orthrus: multi-task linear autoencoder. A shared linear encoder maps the input
to a d-dimensional latent space, from which a linear decoder reconstructs the
input and a linear classification head predicts the response (logit).
"""

import torch.nn as nn


class LinearAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent     = nn.Linear(input_dim, latent_dim)    # encoder
        self.decoder    = nn.Linear(latent_dim, input_dim)    # reconstruction head
        self.classifier = nn.Linear(latent_dim, 1)            # classification head

    def encode(self, x):
        return self.latent(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), self.classifier(z)

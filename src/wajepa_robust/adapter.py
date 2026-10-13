"""Experimental low-rank history-token restoration, trained by clean teacher context.

This is a hypothesis to test, not a claimed novel architecture or proven result.
Clean inputs take an exact identity path. Missing/stale camera tokens receive a
residual conditioned on local features, pooled healthy context, camera and age.
"""
import torch
from torch import nn


class ReliabilityAdapter(nn.Module):
    def __init__(self, dim=512, rank=32, use_age=True):
        super().__init__()
        self.use_age = use_age
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.context = nn.Linear(dim, rank)
        self.camera = nn.Embedding(4, rank)
        self.metadata = nn.Sequential(nn.Linear(2, rank), nn.SiLU(), nn.Linear(rank, rank))
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens, status):
        b, n, d = tokens.shape
        if n != 4096 or status.shape != (b, 4, 4, 2):
            raise ValueError("Expected 4 cameras x 2 tubelets x 512 spatial tokens.")
        # Encoder tokens are camera-major, then temporal, then spatial.
        x = tokens.reshape(b, 4, 2, 512, d)
        meta = status.reshape(b, 2, 2, 4, 2).mean(2).permute(0, 2, 1, 3)
        gate = ((meta[..., 0] < 1) | (meta[..., 1] > 0)).float()
        good = (1 - gate)[..., None, None]
        healthy = (x * good).sum((1, 2, 3)) / (good.sum((1, 2, 3)) * 512).clamp_min(1)
        conditioning = meta.clone()
        if not self.use_age:
            # Generic fault-awareness ablation: availability + binary fault flag.
            conditioning[..., 1] = gate
        hidden = self.down(self.norm(x))
        hidden = hidden + self.context(healthy)[:, None, None, None, :]
        hidden = hidden + self.camera.weight[None, :, None, None, :]
        hidden = hidden + self.metadata(conditioning)[:, :, :, None, :]
        delta = self.up(torch.nn.functional.silu(hidden))
        return (x + gate[..., None, None] * delta).reshape(b, n, d)

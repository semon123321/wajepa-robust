"""Non-learned controls: isolate the value of explicit sensor reliability.

These change history LATENTS, not raw pixel fill or the flow architecture.
They must be compared before attributing gains to a trained adapter.
"""
import torch
from torch import nn


class ReliabilityControl(nn.Module):
    def __init__(self, mode="zero", age_scale_s=0.5):
        super().__init__()
        if mode not in ("zero", "healthy_mean"):
            raise ValueError(mode)
        self.mode = mode
        self.age_scale_s = age_scale_s

    def forward(self, tokens, status):
        b,n,d = tokens.shape
        x = tokens.reshape(b,4,2,512,d)
        meta = status.reshape(b,2,2,4,2).mean(2).permute(0,2,1,3)
        reliability = meta[...,0] * torch.exp(-meta[...,1]/self.age_scale_s)
        if torch.all(reliability==1):
            return tokens
        if self.mode=="zero":
            replacement = torch.zeros_like(x)
        else:
            healthy = (reliability==1)[...,None,None]
            mean = (x*healthy).sum((1,2,3)) / (healthy.sum((1,2,3))*512).clamp_min(1)
            replacement = mean[:,None,None,None,:]
        alpha = reliability[...,None,None]
        return (alpha*x+(1-alpha)*replacement).reshape(b,n,d)

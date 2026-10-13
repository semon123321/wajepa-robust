"""Development-selected routing based only on observable sensor metadata.

Hypothesis: blackout, continuously stale history and burst loss need different
interventions. This hand-written routing is not a learned fault detector and
must be evaluated on held-out logs and unseen fault patterns.
"""
import torch
from torch import nn
from .adapter import ReliabilityAdapter
from .baselines import ReliabilityControl


class HybridAdapter(nn.Module):
    def __init__(self, context_architecture, action_architecture):
        super().__init__()
        self.context = ReliabilityAdapter(**context_architecture)
        self.action = ReliabilityAdapter(**action_architecture)
        self.control = ReliabilityControl("zero", age_scale_s=0.5)

    def forward(self, tokens, status):
        if tokens.size(0)!=1:
            raise ValueError("Research routing currently supports batch size 1.")
        available=status[...,0];age=status[...,1]
        if torch.any(available<1):
            return self.action(tokens,status)
        if not torch.any(age>0):
            return tokens
        early_current=(age[:,:2]==0).all(dim=1)
        late_stale=(age[:,2:]>0).any(dim=1)
        if torch.any(early_current & late_stale):
            return self.control(tokens,status)
        return self.context(tokens,status)

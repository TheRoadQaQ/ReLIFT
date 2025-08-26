import numpy as np
import torch
from collections import defaultdict
import verl.utils.torch_functional as verl_F

def compute_sft_loss(log_prob, eos_mask, entropy, sft_mask):
    sft_losses = -1 * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask & sft_mask)
    return {
        "sft_loss": sft_loss,
    }   

def compute_sft_loss_v1(log_prob, eos_mask, entropy, sft_mask):
    prob = torch.exp(log_prob).detach()
    weight = prob * (1-prob)
    
    sft_losses = -1 * weight * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask & sft_mask)
    return {
        "sft_loss": sft_loss,
    }   

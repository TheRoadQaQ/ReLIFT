import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_sft_loss(log_prob, eos_mask):
    sft_losses = -1 * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

# lower entropy tokens with higher weight
def compute_sft_loss_v1(log_prob, eos_mask, entropy):
    sft_losses = -1 * log_prob
    weight = 0.5 * torch.exp(-entropy.detach())
    sft_losses = weight * sft_losses
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }  

# luffy reshape
def compute_sft_loss_v2(log_prob, eos_mask):
    prob = torch.exp(log_prob)
    shaped_prob = prob/(prob + 0.1)
    
    sft_losses = -1 * shaped_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

# higher entropy tokens with higher weight
def compute_sft_loss_v3(log_prob, eos_mask, entropy):
    sft_losses = -1 * log_prob
    weight = 0.5 * torch.exp(entropy.detach())
    sft_losses = weight * sft_losses
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }  

# only update tokens with lower entropy
def compute_sft_loss_v4(log_prob, eos_mask, entropy, ratio=0.2):
    sft_losses = -1 * log_prob

    masked_entropy = entropy.clone()
    masked_entropy[~eos_mask] = float('inf')

    flat_entropy = masked_entropy.view(-1)
    num_valid = eos_mask.sum().item()
    k = max(1, int(num_valid * ratio))

    topk_entropy, _ = torch.topk(flat_entropy, k, largest=False)
    threshold = topk_entropy[-1]

    # 构造mask: 只有entropy <= threshold的位置为True
    selected_mask = (masked_entropy <= threshold) & eos_mask

    # 只对selected_mask为True的位置计算loss
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

# only update tokens with high entropy
def compute_sft_loss_v5(log_prob, eos_mask, entropy, ratio=0.2):
    sft_losses = -1 * log_prob

    masked_entropy = entropy.clone()
    masked_entropy[~eos_mask] = float('-inf')

    flat_entropy = masked_entropy.view(-1)
    num_valid = eos_mask.sum().item()
    k = max(1, int(num_valid * ratio))  # 至少选1个

    topk_entropy, _ = torch.topk(flat_entropy, k, largest=True)
    threshold = topk_entropy[-1]

    selected_mask = (masked_entropy >= threshold) & eos_mask

    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }
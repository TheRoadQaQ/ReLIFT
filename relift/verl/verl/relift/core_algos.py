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

# when entropy larger then
# weight to each token: w_SFT = 0.5 ∗ exp(−H(t))
# high entropy with lower weight
def compute_sft_loss_v1(log_prob, eos_mask, entropy):
    sft_losses = -1 * log_prob
    weight = 0.5 * torch.exp(-entropy.detach())
    sft_losses = weight * sft_losses
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }  

def compute_sft_loss_v2(log_prob, eos_mask):
    prob = torch.exp(log_prob)
    shaped_prob = prob/(prob + 0.1)
    
    sft_losses = -1 * shaped_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

# when entropy larger then
# weight to each token: w_SFT = 0.5 ∗ exp(−H(t))
# higher entropy with higher weight
def compute_sft_loss_v3(log_prob, eos_mask, entropy):
    sft_losses = -1 * log_prob
    weight = 0.5 * torch.exp(entropy.detach())
    sft_losses = weight * sft_losses
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }  

# only update tokens with lower entropy
def compute_sft_loss_v4(log_prob, eos_mask, entropy, ratio=0.5):
    sft_losses = -1 * log_prob

    # 只选取entropy最小的前20% token
    # entropy: (batch, seq_len)
    # 先把eos_mask为False的地方设为无穷大，避免pad token被选中
    masked_entropy = entropy.clone()
    masked_entropy[~eos_mask] = float('inf')

    # 展平，方便排序
    flat_entropy = masked_entropy.view(-1)
    num_valid = eos_mask.sum().item()
    k = max(1, int(num_valid * ratio))  # 至少选1个

    # 找到前k小的entropy的阈值
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

    # 先把eos_mask为False的地方设为无穷小，避免pad token被选中
    masked_entropy = entropy.clone()
    masked_entropy[~eos_mask] = float('-inf')

    # 展平，方便排序
    flat_entropy = masked_entropy.view(-1)
    num_valid = eos_mask.sum().item()
    k = max(1, int(num_valid * ratio))  # 至少选1个

    # 找到前k大的entropy的阈值
    topk_entropy, _ = torch.topk(flat_entropy, k, largest=True)
    threshold = topk_entropy[-1]

    # 构造mask: 只有entropy >= threshold的位置为True
    selected_mask = (masked_entropy >= threshold) & eos_mask

    # 只对selected_mask为True的位置计算loss
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

# select tokens with mid entropy
def compute_sft_loss_v6(log_prob, eos_mask, entropy, low_ratio=0.25, high_ratio=0.75):
    sft_losses = -log_prob  # [batch, seq_len]

    valid_entropy = entropy[eos_mask]  # 1D tensor, 只包含有效token
    num_valid = valid_entropy.numel()
    if num_valid == 0:
        return {"sft_loss": torch.tensor(0.0, device=log_prob.device)}

    # 计算分位数对应的k
    k_low = max(1, int(num_valid * low_ratio))
    k_high = max(1, int(num_valid * high_ratio))

    # 用topk找分位数阈值
    threshold_low = torch.topk(valid_entropy, k_low, largest=False)[-1]
    threshold_high = torch.topk(valid_entropy, k_high, largest=False)[-1]

    # 构造mask: 只选取25%~75%分位数之间的token，并且是有效token
    selected_mask = (entropy > threshold_low) & (entropy <= threshold_high) & eos_mask  # [batch, seq_len]

    # 只对selected_mask为True的位置计算loss
    sft_loss = masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

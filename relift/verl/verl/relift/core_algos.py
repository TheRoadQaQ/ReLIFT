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
def compute_sft_loss_v4(log_prob, eos_mask, entropy, ratio=0.5):
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


def compute_sft_loss_v6(log_prob, eos_mask, entropy, low_ratio=0.25, high_ratio=0.75):
    """
    仅对 entropy 位于 [low_ratio, high_ratio] 区间内的 EOS token 计算 SFT loss。
    使用两次 torch.topk（升序）获取 25% 与 75% 阈值，避免 torch.sort。
    """
    sft_losses = -log_prob                                # [B, L]

    # 1. 取出所有有效 token 的熵
    eos_mask = eos_mask.bool()
    valid_entropy = entropy[eos_mask]                    # [N]

    N = valid_entropy.numel()
    if N == 0:
        sft_loss = torch.tensor(0.0, device=log_prob.device, requires_grad=True)
        return {"sft_loss": sft_loss}

    # 2. 计算 25% 与 75% 的阈值（升序 topk）
    k_low  = max(1, int(N * low_ratio))
    k_high = max(1, int(N * high_ratio))

    # 升序 topk 取第 k_low 个（largest=False）
    _, idx_low = torch.topk(valid_entropy, k_low, largest=False)
    low_th = valid_entropy[idx_low[-1]]

    # 升序 topk 取第 k_high 个
    _, idx_high = torch.topk(valid_entropy, k_high, largest=False)
    high_th = valid_entropy[idx_high[-1]]

    # 3. 构造掩码
    selected_mask = (entropy >= low_th) & (entropy <= high_th) & eos_mask

    # 4. 计算平均 loss
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {"sft_loss": sft_loss}

# only update tokens with lower entropy but per sentence
def compute_sft_loss_v4_per_sentence(log_prob, eos_mask, entropy, ratio=0.2):
    sft_losses = -1 * log_prob  # [B, T]

    B, T = entropy.shape
    selected_mask = torch.zeros_like(entropy, dtype=torch.bool)

    for i in range(B):
        eos_i = eos_mask[i].bool()  # [T]
        ent_i = entropy[i][eos_i]  # 只保留有效token的entropy
        num_valid = ent_i.numel()
        k = max(1, int(num_valid * ratio))
        topk_ent, _ = torch.topk(ent_i, k, largest=False)
        threshold = topk_ent[-1]

        # 构造该句子的mask
        mask_i = (entropy[i] <= threshold) & eos_i
        selected_mask[i] = mask_i

    # 只对selected_mask为True的位置计算loss
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

# only update tokens with high entropy but per sentence
def compute_sft_loss_v5_per_sentence(log_prob, eos_mask, entropy, ratio=0.2):
    sft_losses = -1 * log_prob  # [B, T]

    B, T = entropy.shape
    selected_mask = torch.zeros_like(entropy, dtype=torch.bool)

    for i in range(B):
        # 获取第 i 个句子的有效 token
        eos_i = eos_mask[i].bool()  # [T]
        # 只保留有效 token 的 entropy
        ent_i = entropy[i][eos_i]

        num_valid = ent_i.numel()
        # 至少选择一个 token
        k = max(1, int(num_valid * ratio))

        # 找到当前句子中 top-k 高熵的 token
        topk_ent, _ = torch.topk(ent_i, k, largest=True)
        # 获取 top-k 中最小的熵值作为阈值
        threshold = topk_ent[-1]

        # 构造该句子的选择掩码
        # 选择熵值大于等于阈值的 token，且这些 token 必须是有效的
        mask_i = (entropy[i] >= threshold) & eos_i
        selected_mask[i] = mask_i

    # 只对 selected_mask 为 True 的位置计算损失
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

def compute_sft_loss_v7(log_prob, eos_mask):
    prob = torch.exp(log_prob).detach()
    sft_losses = -1 * prob * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

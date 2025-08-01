import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_contrastive_sft_loss(target_log_prob, target_eos_mask, fail_log_prob, fail_eos_mask):
    sft_losses = -1 * target_log_prob
    sft_loss = verl_F.masked_mean(sft_losses, target_eos_mask)

    fail_losses = 1 * fail_log_prob
    fail_loss = verl_F.masked_mean(fail_losses, fail_eos_mask)

    contrastive_loss = 7/8 * sft_loss + 1/8 * fail_loss

    return {
        "sft_loss": sft_loss,
        "fail_loss": fail_loss,
        "contrastive_loss": contrastive_loss,
    }   

def compute_dpo_loss(
    target_log_prob, target_eos_mask,
    fail_log_prob, fail_eos_mask
):
    # 计算每个样本的平均 log prob
    chosen_log_prob = verl_F.masked_mean(target_log_prob, target_eos_mask, dim=1)  # [batch]
    rejected_log_prob = verl_F.masked_mean(fail_log_prob, fail_eos_mask, dim=1)    # [batch]

    # DPO 损失
    diff = chosen_log_prob - rejected_log_prob  # [batch]
    dpo_loss = -torch.log(torch.sigmoid(0.1 * diff)).mean()

    return {
        "dpo_loss": dpo_loss,
        "chosen_log_prob": chosen_log_prob.mean(),
        "rejected_log_prob": rejected_log_prob.mean(),
    }
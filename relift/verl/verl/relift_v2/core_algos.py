import numpy as np
import torch
from collections import defaultdict
from verl.trainer.ppo.core_algos import agg_loss
import verl.utils.torch_functional as verl_F

def compute_rl_sft_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    sft_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    alpha=0.5
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        sft_mask (torch.Tensor):
            Mask indicating which Sequence to include in the SFT loss not in RL loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    rl_mask = response_mask & ~sft_mask
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=rl_mask, loss_agg_mode=loss_agg_mode)

    sft_log_prob = log_prob[sft_mask]
    sft_response_mask = response_mask[sft_mask]
    sft_loss = agg_loss(loss_mat=sft_log_prob, loss_mask=sft_response_mask, loss_agg_mode=loss_agg_mode)

    loss = pg_loss + alpha * sft_loss

    return loss, sft_loss, pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


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
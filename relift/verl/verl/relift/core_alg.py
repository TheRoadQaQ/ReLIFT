import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    clip_upper_bound,
    all_max_clip=None, 
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability

    #breakpoint()
    
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    
    ratio = torch.exp(negative_approx_kl) # [bsz, l]

    on_pg_losses = -advantages * ratio
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)
    if upper_bound == clip_upper_bound:
        print('clip upper bound is used: ', clip_upper_bound)

    if loss_remove_clip is False:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)    
    
    pg_losses = on_pg_losses
    
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        #print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "on_pg_loss": on_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "on_policy_prob": on_policy_prob
    }

def compute_sft_loss(log_prob, eos_mask):
    sft_losses = -1 * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_contrastive_sft_loss(target_log_prob, target_eos_mask, fail_log_prob, fail_eos_mask):
    sft_losses = -1 * target_log_prob
    sft_loss = verl_F.masked_mean(sft_losses, target_eos_mask)

    fail_losses = 1 * fail_log_prob
    fail_loss = verl_F.masked_mean(fail_losses, fail_eos_mask)

    contrastive_loss = sft_loss + fail_loss

    return {
        "sft_loss": sft_loss,
        "fail_loss": fail_loss,
        "contrastive_loss": contrastive_loss,
    }   
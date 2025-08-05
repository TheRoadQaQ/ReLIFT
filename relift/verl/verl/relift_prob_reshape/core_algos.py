import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_prob_reshape_sft_loss(log_prob, target_id, eos_mask, tau=0.1):
    '''
    根据提供的公式计算重塑后的目标概率，用于SFT loss。

    Args:
        log_prob (torch.Tensor): 模型的原始log概率输出，形状为 (b x l x vocab_size)。
        target_id (torch.Tensor): 目标token的ID，形状为 (b x l)。
        eos_mask (torch.Tensor): EOS或padding位置的掩码，形状为 (b x l)。值为1的位置保留，为0的位置忽略。
        tau (float): 目标边际 τ。

    Returns:
        torch.Tensor: 重塑后的概率分布 (reshaped_prob)，形状为 (b x l x vocab_size)。
    '''
    # 将log概率转换为概率分布 p^T
    prob = torch.exp(log_prob).detach()
    b, l, v = prob.shape

    # 1. 找到模型预测的最大概率 p_argmax^T 及其对应的 token ID
    p_argmax, argmax_id = torch.max(prob, dim=-1) # (b, l)

    # 2. 找到目标token l 对应的概率 p_l^T
    # torch.gather 用于根据 target_id 在 vocab_size 维度上收集概率值
    p_target = torch.gather(prob, -1, target_id.unsqueeze(-1)).squeeze(-1) # (b, l)

    # 3. 判断模型的预测是否正确 (argmax p^T == l)
    is_correct = (argmax_id == target_id)
    is_incorrect = ~is_correct

    # 4. 根据公式计算两种情况下的缩放因子 (α 和 β)

    # 情况一: 预测错误 (argmax p^T != l), 计算 α
    # alpha = (p_argmax^T - p_l^T + τ) / (1 + p_argmax^T - p_l^T)
    # 为避免分母为零，在分母上增加一个极小值 epsilon
    denominator_alpha = 1 + p_argmax - p_target
    alpha = (p_argmax - p_target + tau) / (denominator_alpha + 1e-9)

    # 情况二: 预测正确 (argmax p^T == l), 计算 β
    # beta = (min(1, p_l^T + τ) - p_l^T) / (1 - p_l^T)
    # 当 p_l^T 趋近于1时，分母为0，此时分子也为0，beta应为0。
    # 我们使用 torch.where 来处理 p_l^T = 1 的情况。
    new_p_target = torch.min(torch.ones_like(p_target), p_target + tau)
    denominator_beta = 1 - p_target
    beta = torch.where(
        denominator_beta > 1e-9,
        (new_p_target - p_target) / denominator_beta,
        torch.zeros_like(denominator_beta) # 如果分母为0，则beta为0
    )

    # 5. 根据预测是否正确，选择对应的因子 (α 或 β)
    factor = torch.where(is_incorrect, alpha, beta) # (b, l)

    # 6. 应用掩码(eos_mask)，在padding或eos位置不进行概率重塑
    # 将掩码为0位置的factor置为0，这样在计算reshaped_prob时，这些位置的prob将保持不变
    factor = factor * eos_mask

    # 7. 准备公式 p^C = (1 - α)p^T + α * 1_l 中的各个部分
    # 将factor的维度从 (b, l) 扩展到 (b, l, 1) 以便进行广播(broadcasting)
    factor = factor.unsqueeze(-1) # (b, l, 1)

    # 创建目标token l 的 one-hot 向量 1_l
    one_hot_target = F.one_hot(target_id, num_classes=v).float() # (b, l, v)

    # 8. 计算最终的 reshaped_prob (即 p^C)
    # p^C = (1 - factor) * p^T + factor * 1_l
    reshaped_prob = (1 - factor) * prob + factor * one_hot_target
    reshaped_prob = reshaped_prob.detach()

    # kl between reshaped_prob and prob
    kl_losses = F.kl_div(input=log_prob, target=reshaped_prob, reduction='none', log_target=False)

    # 对kl_losses求和，得到每个token的损失
    kl_losses = kl_losses.sum(dim=-1) # 形状为 (b, l)

    sft_loss = verl_F.masked_mean(kl_losses, eos_mask)

    return {
        "sft_loss": sft_loss
    }   
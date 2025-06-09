# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.seqlen_balancing import rearrange_micro_batches
import verl.utils.torch_functional as verl_F

__all__ = ['ReLIFTDataParallelPPOActor']

from verl.workers.actor.dp_actor import DataParallelPPOActor

class ReLIFTDataParallelPPOActor(DataParallelPPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        actor_sft_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.use_adaptive_temperature = self.config.use_adaptive_temperature
        self.adaptive_temperature_target_entropy = self.config.adaptive_temperature_target_entropy
        if self.use_adaptive_temperature:
            self.log_alpha = torch.tensor(np.log(self.config.entropy_coeff), dtype=torch.float)
            self.log_alpha.requires_grad = True
            from torch import optim
            self.alpha_optimizer = optim.AdamW([self.log_alpha],
                                          lr=self.config.alpha_lr,
                                          betas=(0.9, 0.999),
                                          weight_decay=1e-2)
        else:
            self.alpha_optimizer = None
        self.actor_sft_optimizer = actor_sft_optimizer
            
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')

        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()
                if self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad()

                for data in micro_batches:
                    print("RL MICROBATCH STEP")
                    data = data.cuda()  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    
                    from .core_alg import compute_token_on_off_policy_loss
                    loss_fn = compute_token_on_off_policy_loss

                    ret_dict = loss_fn(
                        old_log_prob=old_log_prob, 
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=clip_ratio,
                        clip_upper_bound=self.config.clip_upper_bound,
                        all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                        loss_remove_token_mean=self.config.loss_remove_token_mean,
                        loss_remove_clip=self.config.loss_remove_clip
                    )
                    pg_loss = ret_dict['pg_loss']
                    on_pg_loss = ret_dict['on_pg_loss']
                    pg_clipfrac = ret_dict['on_pg_clipfrac']
                    ppo_kl = ret_dict['ppo_kl']
                    
                    data = {
                        'actor/on_pg_loss': on_pg_loss.detach().item(),
                    }
                    if 'on_policy_prob' in ret_dict:
                        data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                    append_to_dict(metrics, data)
                    
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    if self.config.use_adaptive_temperature:
                        if self.config.use_adaptive_temperature_fixed is False:
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            entropy_coeff = self.log_alpha.exp()
                            if self.config.adaptive_temperature_clip > 0:
                                entropy_coeff = torch.clamp(entropy_coeff, max=self.config.adaptive_temperature_clip)
                            alpha_loss = verl_F.masked_mean(entropy - target_entropy, response_mask).detach() * entropy_coeff
                            alpha_loss = alpha_loss / self.gradient_accumulation
                            alpha_loss.backward()
                            
                            policy_loss = pg_loss - entropy_loss * entropy_coeff.detach().item()
                            metrics['actor/alpha_loss'] = alpha_loss.detach().item()
                            metrics['actor/entropy_coeff'] = entropy_coeff.detach().item()
                            metrics['actor/log_alpha'] = self.log_alpha.detach().item()
                        else: # fixed strategy for entropy coeff
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            # cur_entropy = verl_F.masked_mean(entropy, response_mask)
                            entropy_coeff = (target_entropy / entropy_loss).detach().item() * self.config.entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                            metrics['actor/entropy_coeff'] = entropy_coeff
                    else:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef
                    if self.config.use_ppo_kl_loss:
                        policy_loss = policy_loss + ppo_kl.abs() * self.config.ppo_kl_loss_coef
                        metrics['actor/ppo_kl_loss'] = ppo_kl.abs().detach().item()
                        
                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': float(entropy_loss.detach().item()),
                        'actor/pg_loss': float(pg_loss.detach().item()),
                        'actor/pg_clipfrac': float(pg_clipfrac.detach().item()),
                        'actor/ppo_kl': float(ppo_kl.detach().item()),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': float(grad_norm.detach().item())}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        return metrics
    
    def sft_update_policy(self, data: DataProto):
        #breakpoint()

        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.sft.sft_data_size % self.config.sft.sft_mini_batch_size == 0
        assert self.config.sft.sft_mini_batch_size % self.config.sft.sft_micro_batch_size == 0

        self.gradient_accumulation = self.config.sft.sft_mini_batch_size // self.config.sft.sft_micro_batch_size

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'responses']
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.sft.sft_mini_batch_size)

        metrics = {}
        for _ in range(self.config.sft.sft_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                micro_batches = mini_batch.split(self.config.sft.sft_micro_batch_size)

                self.actor_sft_optimizer.zero_grad()

                for data in micro_batches:
                    print("SFT MICROBATCH STEP")
                    data = data.cuda()  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]

                    entropy_coeff = self.config.sft.entropy_coeff

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=1.0)

                    from .core_alg import compute_sft_loss
                    loss_fn = compute_sft_loss
                    ret_dict = loss_fn(log_prob=log_prob, eos_mask=response_mask)
                    sft_loss = ret_dict["sft_loss"]
                
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    loss = sft_loss - entropy_loss * entropy_coeff
                        
                    loss = loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/sft_entropy_loss': float(entropy_loss.detach().cpu().item()),
                        'actor/sft_loss': float(sft_loss.detach().cpu().item())
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._sft_optimizer_step()
                data = {'actor/sft_grad_norm': float(grad_norm.detach().item())}
                #del grad_norm
                
                append_to_dict(metrics, data)
                
        self.actor_sft_optimizer.zero_grad()
        
        return metrics
    
    def _sft_optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_sft_optimizer.step()
        
        return grad_norm

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        
        return grad_norm
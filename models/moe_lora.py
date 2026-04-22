"""LoRA MoE layer following the paper logic closely."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.normal import Normal

from models.gating import SparseDispatcher
from utils.config import GatingConfig, LoRAExpertConfig


@dataclass(slots=True)
class LoRAAuxiliaryOutput:
    importance: Tensor
    load: Tensor


class MoELoRALayer(nn.Module):
    """Token-level top-1 noisy-gated LoRA MoE matching the paper implementation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        experts: list[LoRAExpertConfig],
        gating_config: GatingConfig,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_configs = experts
        self.gating_config = gating_config
        self.num_experts = len(experts)
        self.k = gating_config.top_k
        self.noisy_gating = gating_config.noisy_gating

        self.lora_a = nn.ModuleList()
        self.lora_b = nn.ModuleList()
        for expert in experts:
            lora_a = nn.Linear(input_dim, expert.rank, bias=False)
            nn.init.kaiming_uniform_(lora_a.weight, a=5 ** 0.5)
            lora_b = nn.Linear(expert.rank, output_dim * 3, bias=False)
            nn.init.zeros_(lora_b.weight)
            self.lora_a.append(lora_a)
            self.lora_b.append(lora_b)

        self.w_gate = nn.Parameter(torch.zeros(input_dim, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, self.num_experts), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        if self.k > self.num_experts:
            raise ValueError(f"top-k={self.k} cannot exceed num_experts={self.num_experts}")

    def _empty_aux(self, tokens: Tensor) -> LoRAAuxiliaryOutput:
        zeros = torch.zeros(self.num_experts, device=tokens.device, dtype=tokens.dtype)
        return LoRAAuxiliaryOutput(importance=zeros, load=zeros)

    @staticmethod
    def cv_squared(values: Tensor) -> Tensor:
        eps = 1e-10
        if values.shape[0] == 1:
            return torch.tensor([0], device=values.device, dtype=values.dtype)
        return values.float().var() / (values.float().mean() ** 2 + eps)

    @staticmethod
    def _gates_to_load(gates: Tensor) -> Tensor:
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self,
        clean_values: Tensor,
        noisy_values: Tensor,
        noise_stddev: Tensor,
        noisy_top_values: Tensor,
    ) -> Tensor:
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.gather(top_values_flat, 0, threshold_positions_if_in).unsqueeze(1)
        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.gather(top_values_flat, 0, threshold_positions_if_out).unsqueeze(1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x: Tensor, train: bool, noise_epsilon: float | None = None) -> tuple[Tensor, Tensor]:
        epsilon = self.gating_config.noise_epsilon if noise_epsilon is None else noise_epsilon
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            noise_stddev = None
            noisy_logits = clean_logits
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train and noise_stddev is not None:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, tokens: Tensor, router_enabled: bool = True) -> tuple[Tensor, LoRAAuxiliaryOutput]:
        if not router_enabled:
            qkv_delta = F.linear(tokens, self.lora_a[0].weight)
            qkv_delta = F.linear(qkv_delta, self.lora_b[0].weight)
            return qkv_delta, self._empty_aux(tokens)

        batch_size, num_tokens, hidden_dim = tokens.shape
        flat_tokens = tokens.reshape(batch_size * num_tokens, hidden_dim)
        gates, load = self.noisy_top_k_gating(flat_tokens, self.training)
        importance = gates.sum(0)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(flat_tokens)
        expert_outputs: list[Tensor] = []
        for expert_index in range(self.num_experts):
            if expert_inputs[expert_index].numel() == 0:
                continue
            qkv_delta = F.linear(expert_inputs[expert_index], self.lora_a[expert_index].weight)
            qkv_delta = F.linear(qkv_delta, self.lora_b[expert_index].weight)
            expert_outputs.append(qkv_delta)

        qkv_delta = dispatcher.combine(expert_outputs)
        qkv_delta = qkv_delta.reshape(batch_size, num_tokens, hidden_dim * 3)
        return qkv_delta, LoRAAuxiliaryOutput(importance=importance, load=load)

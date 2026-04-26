"""Sparse dispatch helper matching the paper-style MoE routing logic."""

from __future__ import annotations

import torch
from torch import Tensor


class SparseDispatcher:
    """
    Helper for sparsely-gated MoE execution.

    This follows the paper code closely: inputs are dispatched only to experts
    whose gates are non-zero, and expert outputs are stitched back together via
    `index_add`.
    """

    def __init__(self, num_experts: int, gates: Tensor) -> None:
        self._gates = gates
        self._num_experts = num_experts

        nonzero_gates = torch.nonzero(gates, as_tuple=False)
        if nonzero_gates.numel() == 0:
            self._expert_index = gates.new_zeros((0, 1), dtype=torch.long)
            self._batch_index = gates.new_zeros((0,), dtype=torch.long)
            self._part_sizes = [0 for _ in range(num_experts)]
            self._nonzero_gates = gates.new_zeros((0, 1))
            return

        sorted_experts, index_sorted_experts = nonzero_gates.sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = nonzero_gates[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_expanded = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_expanded, 1, self._expert_index)

    def dispatch(self, inputs: Tensor) -> tuple[Tensor, ...]:
        if self._batch_index.numel() == 0:
            return tuple(inputs.new_empty((0, *inputs.shape[1:])) for _ in range(self._num_experts))
        dispatched = inputs[self._batch_index]
        if dispatched.ndim > 1:
            dispatched = dispatched.squeeze(1)
        return torch.split(dispatched, self._part_sizes, dim=0)

    def combine(self, expert_outputs: list[Tensor], multiply_by_gates: bool = True) -> Tensor:
        if not expert_outputs:
            raise ValueError("expert_outputs must not be empty when combining dispatched outputs.")

        # FIX START
        stitched = torch.cat(expert_outputs, dim=0)
        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates.to(dtype=stitched.dtype)

        zeros = torch.zeros(
            self._gates.size(0),
            expert_outputs[-1].size(1),
            dtype=stitched.dtype,
            device=stitched.device,
        )
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined
        # FIX END

    def expert_to_gates(self) -> tuple[Tensor, ...]:
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

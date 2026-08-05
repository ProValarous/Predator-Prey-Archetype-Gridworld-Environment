# src/ppage/baselines/A3C/shared_adam.py
"""
Adam variant whose per-parameter optimizer state lives in shared memory.

Plain torch.optim.Adam keeps its state (exp_avg, exp_avg_sq, step count) as
regular tensors private to whichever process initializes them. A3C's worker
processes all step the *same* global network's parameters, so the optimizer
state needs to live in shared memory too -- otherwise each process would
silently keep its own separate, diverging view of the Adam moment estimates.
"""

import torch


class SharedAdam(torch.optim.Adam):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()

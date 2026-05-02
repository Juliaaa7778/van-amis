"""
Muon optimizer implementation.

Muon - MomentUm Orthogonalized by Newton-schulz

Original implementation from MoonshotAI/Moonlight:
https://github.com/MoonshotAI/Moonlight

Modified version adapted from:
https://github.com/KellerJordan/Muon/blob/master/muon.py
"""

import math
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    We opt to use a quintic iteration whose coefficients are selected to maximize
    the slope at zero. For the purpose of minimizing steps, it turns out to be
    empirically effective to keep increasing the slope at zero even beyond the point
    where the iteration no longer converges all the way to one everywhere on the interval.

    This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    Args:
        G: Input matrix to orthogonalize.
        steps: Number of Newton-Schulz iterations.

    Returns:
        Orthogonalized matrix.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the
    nearest orthogonal matrix. To efficiently orthogonalize each update, we use a
    Newton-Schulz iteration, which has the advantage that it can be stably run in
    bfloat16 on the GPU.

    Warnings:
        - This optimizer is unlikely to work well for training with small batch size.
        - It may not work well for finetuning pretrained models.

    Args:
        lr: The learning rate. The updates will have spectral norm of `lr`.
            (0.02 is a good default)
        wd: Weight decay coefficient.
        muon_params: The parameters to be optimized by Muon.
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        ns_steps: The number of Newton-Schulz iterations to run.
            (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in
            `muon_params` which are {0, 1}-D or are detected as being the embed
            or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.1,
        muon_params: Optional[Iterable[Tensor]] = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[Iterable[Tensor]] = None,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []

        params = muon_params + adamw_params
        super().__init__(params, defaults)

        # Mark which parameters use Muon vs AdamW
        for p in muon_params:
            assert p.ndim == 2, f"Muon params must be 2D, got {p.ndim}D"
            self.state[p]["use_muon"] = True

        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr: float, param_shape: tuple) -> float:
        """
        Adjust learning rate based on parameter matrix size.

        As described in the paper, we adjust the learning rate and weight decay
        based on the size of the parameter matrix.

        Args:
            lr: Base learning rate.
            param_shape: Shape of the parameter tensor.

        Returns:
            Adjusted learning rate.
        """
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon update for 2D parameters
            self._muon_step(group)

            # AdamW update for other parameters
            self._adamw_step(group)

        return loss

    def _muon_step(self, group: dict) -> None:
        """Perform Muon update step."""
        params = [p for p in group["params"] if self.state[p]["use_muon"]]
        lr = group["lr"]
        wd = group["wd"]
        momentum = group["momentum"]

        for p in params:
            g = p.grad
            if g is None:
                continue

            if g.ndim > 2:
                g = g.view(g.size(0), -1)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)

            if group["nesterov"]:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf

            u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

            adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

            # Apply weight decay
            p.data.mul_(1 - lr * wd)

            # Apply update
            p.data.add_(u, alpha=-adjusted_lr)

    def _adamw_step(self, group: dict) -> None:
        """Perform AdamW update step."""
        params = [p for p in group["params"] if not self.state[p]["use_muon"]]
        lr = group["lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        weight_decay = group["wd"]

        for p in params:
            g = p.grad
            if g is None:
                continue

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["moment1"] = torch.zeros_like(g)
                state["moment2"] = torch.zeros_like(g)

            state["step"] += 1
            step = state["step"]
            buf1 = state["moment1"]
            buf2 = state["moment2"]

            buf1.lerp_(g, 1 - beta1)
            buf2.lerp_(g.square(), 1 - beta2)

            g = buf1 / (eps + buf2.sqrt())

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            scale = bias_correction1 / bias_correction2**0.5

            p.data.mul_(1 - lr * weight_decay)
            p.data.add_(g, alpha=-lr / scale)


def get_optimizer(
    optimizer_name: str,
    model: torch.nn.Module,
    lr: float = 1e-3,
    wd: float = 0.1,
) -> Optimizer:
    """
    Create an optimizer for the given model.

    Args:
        optimizer_name: Name of the optimizer ('adamw', 'muon', 'adam', 'sgd').
        model: The model to optimize.
        lr: Learning rate.
        wd: Weight decay.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If optimizer_name is not supported.
    """
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed" not in name.lower() and "head" not in name.lower()
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed" not in name.lower() and "head" not in name.lower()
            )
        ]
        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        raise ValueError(f"Optimizer not supported: {optimizer_name}")

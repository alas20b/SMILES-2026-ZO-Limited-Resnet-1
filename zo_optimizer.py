"""
zo_optimizer.py – Zero-order optimizer with repeated SPSA sampling per step.
Uses num_estimates_per_step independent SPSA estimates per optimization step
to reduce gradient noise, while staying within total forward pass budget.
"""

import torch
import torch.nn as nn
from typing import Callable


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        eps: float = 0.001,
        num_estimates_per_step: int = 16,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.num_estimates_per_step = num_estimates_per_step

        # Train only the final classification head (fc) – few parameters, stable learning
        self.layer_names = ["fc.weight", "fc.bias"]

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Parameters not found: {missing}")
        return {n: named[n] for n in self.layer_names}

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()

        # Record initial loss (before any perturbation)
        with torch.no_grad():
            loss_before = loss_fn()

        # Accumulator for gradient estimates
        grad_accum = {name: torch.zeros_like(p) for name, p in params.items()}

        for _ in range(self.num_estimates_per_step):
            # Sample random perturbation Δ ~ {+1, -1} for all parameters
            deltas = {}
            for name, p in params.items():
                deltas[name] = torch.randint(
                    0, 2, p.shape, dtype=torch.float32, device=p.device
                ) * 2 - 1

            # f(θ + ε·Δ)
            for name, p in params.items():
                p.data.add_(self.eps * deltas[name])
            loss_plus = loss_fn()

            # f(θ − ε·Δ)
            for name, p in params.items():
                p.data.sub_(2.0 * self.eps * deltas[name])
            loss_minus = loss_fn()

            # Restore original parameters
            for name, p in params.items():
                p.data.copy_(params[name])

            # Gradient estimate for this sample
            grad = (loss_plus - loss_minus) / (2.0 * self.eps)
            for name, delta in deltas.items():
                grad_accum[name] += grad * delta

        # Average gradients
        for name in grad_accum:
            grad_accum[name] /= self.num_estimates_per_step

        # Simple SGD update (no momentum)
        with torch.no_grad():
            for name, param in params.items():
                param.data.sub_(self.lr * grad_accum[name])

        return float(loss_before)
# """
# zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

# Students: Implement your gradient-free optimization logic inside
# ``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
# estimator as a starting point — you are expected to replace or extend it.

# Key design points
# -----------------
# * **Layer selection** is entirely your responsibility. Set ``self.layer_names``
#   to the list of parameter names you want to optimize. You can change this list
#   at any time — even between ``.step()`` calls — to implement curriculum or
#   progressive-layer strategies.
# * **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
#   exactly ``n_batches`` times. Each call may invoke the model as many times as
#   your estimator requires, but be mindful that more evaluations per step leave
#   fewer steps in the total budget.
# * **No gradients** are computed anywhere in this file. All updates must be
#   derived from scalar loss values obtained by calling ``loss_fn()``.
# """

# from __future__ import annotations

# import math
# from typing import Callable

# import torch
# import torch.nn as nn


# class ZeroOrderOptimizer:
#     """Gradient-free optimizer for fine-tuning a subset of model parameters.

#     The optimizer maintains a list of *active* parameter names
#     (``self.layer_names``). On each ``.step()`` call it perturbs only those
#     parameters, estimates a pseudo-gradient from forward-pass loss values, and
#     applies an update. All other parameters remain strictly frozen.

#     Args:
#         model:            The ``nn.Module`` to optimize.
#         lr:               Step size / learning rate.
#         eps:              Perturbation magnitude for the finite-difference
#                           estimator.
#         perturbation_mode: Distribution used to sample the perturbation
#                           direction. ``"gaussian"`` draws from N(0, I);
#                           ``"uniform"`` draws from U(-1, 1) and normalises.

#     Student task:
#         1. Set ``self.layer_names`` to the parameter names you want to tune.
#            Inspect available names with ``[n for n, _ in model.named_parameters()]``.
#         2. Replace or extend ``_estimate_grad`` with a better estimator.
#         3. Replace or extend ``_update_params`` with a better update rule.
#         4. Optionally change ``self.layer_names`` inside ``.step()`` to
#            implement dynamic layer selection strategies.

#     Example — tune only the final linear layer::

#         optimizer = ZeroOrderOptimizer(model)
#         optimizer.layer_names = ["fc.weight", "fc.bias"]
#     """

#     def __init__(
#         self,
#         model: nn.Module,
#         lr: float = 5e-4, #5e-5, 1e-4, 2e-4,
#         eps: float = 0.02, #0.005, 0.05
#         perturbation_mode: str = "gaussian",
#     ) -> None:
#         self.model = model
#         self.lr = lr
#         self.eps = eps

#         if perturbation_mode not in ("gaussian", "uniform"):
#             raise ValueError(
#                 f"perturbation_mode must be 'gaussian' or 'uniform', "
#                 f"got '{perturbation_mode}'"
#             )
#         self.perturbation_mode = perturbation_mode

#         # ------------------------------------------------------------------
#         # STUDENT: Set self.layer_names to the parameters you want to tune.
#         #
#         # The default below selects only the final classification head.
#         # You may replace this with any subset of named parameters, e.g.:
#         #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
#         #
#         # You can also update self.layer_names inside .step() to implement
#         # a dynamic schedule (e.g. gradually unfreeze deeper layers).
#         # ------------------------------------------------------------------
#         # EDIT: Tune all trainable parameters for full fine‑tuning.
#         self.layer_names: list[str] = [name for name, _ in model.named_parameters()]

#         # EDIT: Additional hyperparameters for advanced update rule.
#         self.momentum = 0.9
#         self.beta2 = 0.99 #0.999
#         self.weight_decay = 1e-5 #1e-4, 0
#         self.total_steps = 0 #256          # will be overwritten if needed
#         self.step_count = 0
#         self.momentum_buffers: dict[str, torch.Tensor] = {}
#         self.avg_sq: dict[str, torch.Tensor] = {}
#         # ------------------------------------------------------------------

#     # ------------------------------------------------------------------
#     # Internal helpers — students may modify these.
#     # ------------------------------------------------------------------

#     def _active_params(self) -> dict[str, nn.Parameter]:
#         """Return a mapping from name → parameter for all active layer names.

#         Only parameters whose names appear in ``self.layer_names`` are
#         returned. Parameters not in this mapping are never modified.

#         Returns:
#             Dict mapping parameter name to its ``nn.Parameter`` tensor.

#         Raises:
#             KeyError: If a name in ``self.layer_names`` does not exist in the
#                       model.
#         """
#         named = dict(self.model.named_parameters())
#         missing = [n for n in self.layer_names if n not in named]
#         if missing:
#             raise KeyError(
#                 f"The following layer names were not found in the model: "
#                 f"{missing}. Use [n for n, _ in model.named_parameters()] "
#                 f"to inspect valid names."
#             )
#         return {n: named[n] for n in self.layer_names}

#     def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
#         """Sample a random unit-norm perturbation vector of the same shape as ``param``.

#         Args:
#             param: The parameter tensor whose shape determines the output shape.

#         Returns:
#             A tensor of the same shape as ``param``, normalised to unit L2 norm.
#         """
#         if self.perturbation_mode == "gaussian":
#             u = torch.randn_like(param)
#         else:  # uniform
#             u = torch.rand_like(param) * 2.0 - 1.0

#         norm = u.norm()
#         if norm > 0:
#             u = u / norm
#         return u

#     def _estimate_grad(
#         self,
#         loss_fn: Callable[[], float],
#         params: dict[str, nn.Parameter],
#     ) -> dict[str, torch.Tensor]:
#         """Estimate a pseudo-gradient for each active parameter.

#         Skeleton: 2-point central-difference estimator.
#         For each active parameter ``p`` independently:
#             1. Sample a random unit vector ``u`` of the same shape as ``p``.
#             2. Evaluate  f_plus  = loss_fn() with ``p ← p + eps * u``
#             3. Evaluate  f_minus = loss_fn() with ``p ← p - eps * u``
#             4. Restore ``p`` to its original value.
#             5. Pseudo-gradient ← ``(f_plus - f_minus) / (2 * eps) * u``

#         This is an unbiased estimator of the directional derivative along ``u``
#         scaled back to parameter space.

#         Args:
#             loss_fn: Callable that evaluates the objective on the current batch
#                      and returns a scalar ``float``. May be called multiple
#                      times; each call must use the *same* batch.
#             params:  Dict of active parameter name → tensor (from
#                      ``_active_params``).

#         Returns:
#             Dict mapping each parameter name to its estimated pseudo-gradient
#             tensor (same shape as the parameter).

#         Student task:
#             Replace this with a more efficient or accurate estimator:
#         """
#         # ------------------------------------------------------------------
#         # STUDENT: Replace or extend the gradient estimation below.
#         # ------------------------------------------------------------------
#         # EDIT: Implement SPSA (Simultaneous Perturbation Stochastic Approximation)
#         #       – only 2 loss evaluations per step, independent of #params.
#         original_vals = {name: p.clone() for name, p in params.items()}

#         # Generate one random perturbation per parameter: delta ~ {+1, -1}
#         deltas = {}
#         for name, p in params.items():
#             deltas[name] = torch.randint(0, 2, p.shape, dtype=torch.float32, device=p.device) * 2 - 1

#         # f_plus = loss(theta + c * delta)
#         for name, p in params.items():
#             p.data.add_(self.eps * deltas[name])
#         loss_plus = loss_fn()

#         # f_minus = loss(theta - c * delta)
#         for name, p in params.items():
#             p.data.sub_(2.0 * self.eps * deltas[name])
#         loss_minus = loss_fn()

#         # Restore original parameters
#         for name, p in params.items():
#             p.data.copy_(original_vals[name])

#         # Compute gradient estimate for each parameter
#         grads = {}
#         for name, delta in deltas.items():
#             grads[name] = ((loss_plus - loss_minus) / (2.0 * self.eps)) * delta

#         return grads
#         # ------------------------------------------------------------------

#     def _update_params(
#         self,
#         params: dict[str, nn.Parameter],
#         grads: dict[str, torch.Tensor],
#     ) -> None:
#         """Apply the estimated pseudo-gradients to the active parameters.

#         Skeleton: vanilla gradient *descent* step (minimising the loss).
#             ``p ← p - lr * grad``

#         Args:
#             params: Dict of active parameter name → tensor.
#             grads:  Dict of pseudo-gradient name → tensor (same keys as
#                     ``params``).

#         Student task:
#             Replace with a more sophisticated update rule, e.g.:
#               - Momentum: accumulate an exponential moving average of gradients.
#               - Adam-style: maintain first and second moment estimates.
#               - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
#         """
#         # ------------------------------------------------------------------
#         # STUDENT: Replace or extend the parameter update below.
#         # ------------------------------------------------------------------
#         # EDIT: Use momentum + RMSprop adaptive scaling + weight decay + gradient clipping.
#         with torch.no_grad():
#             for name, param in params.items():
#                 if name not in self.momentum_buffers:
#                     self.momentum_buffers[name] = torch.zeros_like(param)
#                     # Initialize avg_sq with small positive value to avoid division by zero
#                     self.avg_sq[name] = torch.full_like(param, 1e-4)

#                 # Gradient clipping (elementwise) to prevent extreme updates
#                 grad_clipped = grads[name].clamp(-0.1, 0.1) #clamp(-1.0, 1.0) .clamp(-5, 5)

#                 # Momentum update
#                 self.momentum_buffers[name] = (
#                     self.momentum * self.momentum_buffers[name] + grad_clipped
#                 )

#                 # RMSprop second moment
#                 self.avg_sq[name] = (
#                     self.beta2 * self.avg_sq[name] + (1 - self.beta2) * (grad_clipped ** 2)
#                 )

#                 # Adaptive step
#                 denom = torch.sqrt(self.avg_sq[name]) + 1e-8
#                 step = self.lr * self.momentum_buffers[name] / denom

#                 param.data.sub_(step)

#                 # Weight decay (L2 regularization)
#                 param.data.sub_(self.lr * self.weight_decay * param)
#         # ------------------------------------------------------------------

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def step(self, loss_fn: Callable[[], float]) -> float:
#         """Perform one zero-order optimisation step.

#         Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
#         the currently active parameters (``self.layer_names``), then applies
#         an update. Parameters *not* in ``self.layer_names`` are never touched.

#         Args:
#             loss_fn: A callable that takes no arguments and returns a scalar
#                      ``float`` representing the loss on the current mini-batch.
#                      ``validate.py`` guarantees that every call to ``loss_fn``
#                      within a single ``.step()`` invocation uses the *same*
#                      fixed batch of data.

#         Returns:
#             The loss value at the *start* of the step (before any update),
#             obtained from the first call to ``loss_fn()``.

#         Note:
#             ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
#             Each forward pass inside ``loss_fn`` counts toward your compute
#             budget, so prefer estimators that minimise the number of calls.
#         """
#         params = self._active_params()

#         # Record the loss before any perturbation.
#         with torch.no_grad():
#             loss_before = loss_fn()

#         # Linear decay of learning rate from initial value to zero
#         # if self.total_steps > 0:
#         #     current_lr = self.lr * (1.0 - self.step_count / self.total_steps)
#         # else:
#             current_lr = self.lr
#         self.step_count += 1
#         # Temporarily store the current_lr for use in _update_params
#         original_lr = self.lr
#         self.lr = current_lr

#         grads = self._estimate_grad(loss_fn, params)
#         self._update_params(params, grads)

#         self.lr = original_lr  # restore original for next step

#         return float(loss_before)
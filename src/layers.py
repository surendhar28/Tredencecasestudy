"""
layers.py
---------
Custom PyTorch layers supporting soft weight-level pruning via learnable gate
scores. Gradients flow through both the weight tensor and gate_scores tensor,
enabling end-to-end training of pruning masks together with network weights.

Key design decisions
--------------------
* gate_scores has the *same shape* as the weight parameter so every individual
  weight can be independently gated.
* The temperature τ controls how "sharp" the sigmoid is:
  - large τ  → smooth, nearly-linear gating (training-friendly early on)
  - small τ  → near-binary mask (good for hard pruning later)
* gate_init controls the starting value of gate_scores:
  - 0.0  → sigmoid ≈ 0.5 (fully uncertain, default)
  - +0.5 → gates slightly open; sparsity penalty can push them to 0 faster,
            producing more aggressive bimodal separation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fully-Connected Prunable Layer
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """Drop-in replacement for nn.Linear with soft weight-level gating.

    Parameters
    ----------
    in_features  : int   – number of input features
    out_features : int   – number of output features
    bias         : bool  – whether to add a learnable bias (default: True)
    init_temp    : float – initial temperature for sigmoid sharpening
    gate_init    : float – initial value for gate_scores (default 0.5)
                           positive → gates start slightly open so the sparsity
                           penalty can drive them to 0 more aggressively.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_temp: float = 1.0,
        gate_init: float = 0.5,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # --- standard parameters ---
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # --- gating parameter (same shape as weight) ---
        # Initialised at gate_init so sigmoid starts near a value that allows
        # the sparsity loss to push gates bimodally (0 = pruned, 1 = active).
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), float(gate_init))
        )

        # temperature (scalar; NOT a nn.Parameter – scheduled externally)
        self.temperature = init_temp

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gated linear activation."""
        gates = torch.sigmoid(self.gate_scores / self.temperature)  # ∈ (0,1)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------
    def get_gates(self) -> torch.Tensor:
        """Return current soft gate values (detached from graph)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores / self.temperature)

    def sparsity(self, threshold: float = 0.01) -> float:
        """Fraction of gates below *threshold* (≈ pruned weights)."""
        g = self.get_gates()
        return (g < threshold).float().mean().item()

    # ------------------------------------------------------------------
    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"temperature={self.temperature:.4f}"
        )


# ---------------------------------------------------------------------------
# Convolutional Prunable Layer
# ---------------------------------------------------------------------------

class PrunableConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d with soft filter-weight gating.

    Each individual kernel weight gets its own learnable gate score so pruning
    operates at weight-level granularity.  Channel-level (structured) pruning
    can be derived post-hoc from the mean gate value per output channel.

    Parameters (mirror nn.Conv2d where appropriate)
    ----------
    in_channels, out_channels, kernel_size, stride, padding, bias
    init_temp : float – initial gating temperature
    gate_init : float – initial gate_scores value (default 0.5)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        init_temp: float = 1.0,
        gate_init: float = 0.5,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        # weight shape: (out_channels, in_channels, kH, kW)
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)

        self.weight      = nn.Parameter(torch.empty(*weight_shape))
        self.bias        = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.gate_scores = nn.Parameter(
            torch.full(weight_shape, float(gate_init))
        )

        self.temperature = init_temp
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores / self.temperature)
        pruned_weight = self.weight * gates
        return F.conv2d(
            x, pruned_weight, self.bias,
            stride=self.stride, padding=self.padding
        )

    # ------------------------------------------------------------------
    def get_gates(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores / self.temperature)

    def sparsity(self, threshold: float = 0.01) -> float:
        g = self.get_gates()
        return (g < threshold).float().mean().item()

    def channel_importance(self) -> torch.Tensor:
        """Mean gate value per output channel → used for neuron-level pruning."""
        g = self.get_gates()            # (out_C, in_C, kH, kW)
        return g.mean(dim=(1, 2, 3))    # (out_C,)

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, temperature={self.temperature:.4f}"
        )

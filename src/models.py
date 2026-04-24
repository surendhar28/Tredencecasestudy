"""
models.py
---------
Four architectures for CIFAR-10 (32×32 RGB input, 10 classes):

  1. BaselineCNN     – standard CNN, no pruning, accuracy reference.
  2. BaselineMLP     – standard MLP (NO convolutions), non-CNN accuracy reference.
  3. SelfPruningCNN  – CNN with PrunableConv2d + PrunableLinear + temperature schedule.
  4. SelfPruningMLP  – MLP with PrunableLinear + temperature schedule.

CNN Architecture (shared between BaselineCNN and SelfPruningCNN)
-----------------------------------------------------------------
  Block 1: Conv(3, 64, 3) → BN → ReLU → Conv(64, 64, 3) → BN → ReLU → MaxPool(2)
  Block 2: Conv(64,128, 3) → BN → ReLU → Conv(128,128,3) → BN → ReLU → MaxPool(2)
  Block 3: Conv(128,256,3) → BN → ReLU → Conv(256,256,3) → BN → ReLU → MaxPool(2)
  Head:    AdaptiveAvgPool(2,2) → Flatten → FC(1024,512) → ReLU → Drop(0.5)
           → FC(512,256) → ReLU → FC(256,10)

MLP Architecture (shared between BaselineMLP and SelfPruningMLP)
-----------------------------------------------------------------
  Flatten(3072) → FC(3072,512) → BN1d → ReLU → Dropout(0.4)
               → FC(512,256)  → BN1d → ReLU
               → FC(256,10)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from src.layers import PrunableLinear, PrunableConv2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conv_block(
    in_ch: int,
    out_ch: int,
    prunable: bool = False,
    init_temp: float = 1.0,
    gate_init: float = 0.5,
) -> nn.Sequential:
    """Two-conv block with BN, ReLU, and MaxPool."""
    if prunable:
        c1 = PrunableConv2d(in_ch,  out_ch, kernel_size=3, padding=1,
                            init_temp=init_temp, gate_init=gate_init)
        c2 = PrunableConv2d(out_ch, out_ch, kernel_size=3, padding=1,
                            init_temp=init_temp, gate_init=gate_init)
    else:
        c1 = nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=True)
        c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)

    return nn.Sequential(
        c1, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        c2, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )


# ---------------------------------------------------------------------------
# 1. Baseline CNN (no pruning)
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """Standard CNN on CIFAR-10 – used as CNN accuracy reference."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _make_conv_block(3,   64,  prunable=False),
            _make_conv_block(64,  128, prunable=False),
            _make_conv_block(128, 256, prunable=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2. Baseline MLP (no pruning, no convolutions)
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """Pure MLP baseline on CIFAR-10 – no convolutions, flattened input only.

    This serves as the non-CNN reference model to compare against both
    BaselineCNN (conv reference) and SelfPruningMLP (pruned MLP).

    Architecture:
        Flatten(3072) → FC(512) → BN → ReLU → Dropout(0.4)
                      → FC(256) → BN → ReLU
                      → FC(10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. Self-Pruning CNN
# ---------------------------------------------------------------------------

class SelfPruningCNN(nn.Module):
    """CNN with soft weight-level self-pruning on FC (+ optionally Conv) layers.

    Parameters
    ----------
    num_classes  : int   – output classes
    prune_conv   : bool  – if True, use PrunableConv2d in convolution blocks
    init_temp    : float – starting temperature for gate sharpness
    dropout      : float – dropout probability in classifier head
    gate_init    : float – initial gate_scores value (0.5 = slightly open;
                           the sparsity penalty then drives them toward 0)
    """

    def __init__(
        self,
        num_classes: int = 10,
        prune_conv:  bool = True,
        init_temp:   float = 1.0,
        dropout:     float = 0.5,
        gate_init:   float = 0.5,
    ):
        super().__init__()
        self.prune_conv = prune_conv
        self._temperature = init_temp
        self._gate_init = gate_init

        self.features = nn.Sequential(
            _make_conv_block(3,   64,  prunable=prune_conv,
                             init_temp=init_temp, gate_init=gate_init),
            _make_conv_block(64,  128, prunable=prune_conv,
                             init_temp=init_temp, gate_init=gate_init),
            _make_conv_block(128, 256, prunable=prune_conv,
                             init_temp=init_temp, gate_init=gate_init),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # All FC layers are always prunable
        self.classifier = nn.Sequential(
            PrunableLinear(256 * 4, 512, init_temp=init_temp, gate_init=gate_init),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            PrunableLinear(512, 256, init_temp=init_temp, gate_init=gate_init),
            nn.ReLU(inplace=True),
            PrunableLinear(256, num_classes, init_temp=init_temp, gate_init=gate_init),
        )

    # ------------------------------------------------------------------
    # Temperature management
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """Propagate new temperature to all prunable sub-layers."""
        self._temperature = value
        for m in self.modules():
            if isinstance(m, (PrunableLinear, PrunableConv2d)):
                m.temperature = value

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def prunable_layers(self) -> list:
        return [
            m for m in self.modules()
            if isinstance(m, (PrunableLinear, PrunableConv2d))
        ]

    def sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity penalty = mean of sigmoid(gate_scores) across all layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores / layer.temperature)
            total = total + gates.mean()
            count += 1
        return total / max(count, 1)

    def global_sparsity(self, threshold: float = 0.01) -> float:
        """Fraction of all gates that are effectively zero (< threshold)."""
        closed, total = 0, 0
        for layer in self.prunable_layers():
            g = layer.get_gates()
            closed += (g < threshold).sum().item()
            total  += g.numel()
        return closed / total if total > 0 else 0.0

    def count_parameters(self, count_gates: bool = False) -> int:
        """Count *weight* parameters (gate_scores excluded by default)."""
        total = 0
        for name, p in self.named_parameters():
            if not count_gates and "gate_scores" in name:
                continue
            if p.requires_grad:
                total += p.numel()
        return total

    def layer_sparsities(self, threshold: float = 0.01) -> dict:
        """Return {layer_name: sparsity_fraction} for each prunable layer."""
        result = {}
        for name, m in self.named_modules():
            if isinstance(m, (PrunableLinear, PrunableConv2d)):
                result[name] = m.sparsity(threshold)
        return result

    def fc_neuron_importance(self) -> list:
        """Per-neuron mean gate scores for every PrunableLinear layer."""
        infos = []
        for name, m in self.named_modules():
            if isinstance(m, PrunableLinear):
                g = m.get_gates()
                importance = g.mean(dim=1)
                infos.append({"name": name, "importance": importance})
        return infos

    def extra_repr(self):
        return (
            f"prune_conv={self.prune_conv}, "
            f"temperature={self._temperature:.4f}, "
            f"gate_init={self._gate_init}"
        )


# ---------------------------------------------------------------------------
# 4. Self-Pruning MLP (non-CNN prunable model)
# ---------------------------------------------------------------------------

class SelfPruningMLP(nn.Module):
    """MLP with soft weight-level self-pruning — no convolutions.

    Prunable version of BaselineMLP. Provides a fair apples-to-apples
    comparison: same FC topology, but all Linear layers replaced with
    PrunableLinear.

    Architecture:
        Flatten(3072) → PrunableLinear(512) → BN → ReLU → Dropout(0.4)
                      → PrunableLinear(256) → BN → ReLU
                      → PrunableLinear(10)

    Parameters
    ----------
    num_classes : int   – output classes
    init_temp   : float – starting gate temperature
    gate_init   : float – initial gate_scores value (0.5 recommended)
    dropout     : float – dropout after first FC
    """

    def __init__(
        self,
        num_classes: int   = 10,
        init_temp:   float = 1.0,
        gate_init:   float = 0.5,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self._temperature = init_temp
        self._gate_init = gate_init
        self.flatten = nn.Flatten()

        self.net = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512, init_temp=init_temp, gate_init=gate_init),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            PrunableLinear(512, 256, init_temp=init_temp, gate_init=gate_init),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            PrunableLinear(256, num_classes, init_temp=init_temp, gate_init=gate_init),
        )

    # ------------------------------------------------------------------
    # Temperature management
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = value
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                m.temperature = value

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))

    # ------------------------------------------------------------------
    # Introspection helpers (duck-typing compatible with SelfPruningCNN)
    # ------------------------------------------------------------------

    def prunable_layers(self) -> list:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity penalty = mean of sigmoid(gate_scores) across all layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores / layer.temperature)
            total = total + gates.mean()
            count += 1
        return total / max(count, 1)

    def global_sparsity(self, threshold: float = 0.01) -> float:
        closed, total = 0, 0
        for layer in self.prunable_layers():
            g = layer.get_gates()
            closed += (g < threshold).sum().item()
            total  += g.numel()
        return closed / total if total > 0 else 0.0

    def count_parameters(self, count_gates: bool = False) -> int:
        total = 0
        for name, p in self.named_parameters():
            if not count_gates and "gate_scores" in name:
                continue
            if p.requires_grad:
                total += p.numel()
        return total

    def layer_sparsities(self, threshold: float = 0.01) -> dict:
        result = {}
        for name, m in self.named_modules():
            if isinstance(m, PrunableLinear):
                result[name] = m.sparsity(threshold)
        return result

    def fc_neuron_importance(self) -> list:
        infos = []
        for name, m in self.named_modules():
            if isinstance(m, PrunableLinear):
                g = m.get_gates()
                importance = g.mean(dim=1)
                infos.append({"name": name, "importance": importance})
        return infos

    def extra_repr(self):
        return (
            f"temperature={self._temperature:.4f}, "
            f"gate_init={self._gate_init}"
        )

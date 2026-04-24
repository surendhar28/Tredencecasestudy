"""
pruning.py
----------
Post-training hard-pruning utilities.

After soft-gate training we "freeze" the pruning decisions:
  1. Convert soft gates → binary masks  (gate > threshold → 1, else → 0)
  2. Inject binary masks as non-trainable buffers in a frozen copy of the model.
  3. Count the *effective* (non-zero) parameters.
  4. Perform neuron-level structured pruning for PrunableLinear layers
     (remove entire output neurons whose mean gate value is below a threshold).
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn

from src.layers import PrunableLinear, PrunableConv2d
from src.models import SelfPruningCNN


# ---------------------------------------------------------------------------
# 1. Apply hard binary masks (weight-level pruning)
# ---------------------------------------------------------------------------

def apply_hard_masks(
    model: SelfPruningCNN,
    threshold: float = 0.5,
) -> SelfPruningCNN:
    """
    Return a deep-copy of *model* with gate_scores replaced by hard binary masks.

    Any gate value ≤ threshold is zeroed out permanently:
        new_weight = weight * (sigmoid(gate_scores / T) > threshold)

    The resulting model has the same architecture but masked weights.
    The gate_scores parameter is set to a large constant so sigmoid → 1
    and the mask is embedded in the weight itself.
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    with torch.no_grad():
        for m in model_copy.modules():
            if isinstance(m, (PrunableLinear, PrunableConv2d)):
                gates = torch.sigmoid(m.gate_scores / m.temperature)
                mask  = (gates > threshold).float()
                # Bake mask into weight → zero irrelevant weights
                m.weight.data.mul_(mask)
                # Set gate_scores to large positive → sigmoid → 1 (transparent)
                m.gate_scores.data.fill_(10.0)

    return model_copy


# ---------------------------------------------------------------------------
# 2. Effective parameter count
# ---------------------------------------------------------------------------

def count_effective_params(model: SelfPruningCNN, threshold: float = 0.5) -> int:
    """
    Count non-zero weight entries over all prunable layers (after hard masking).
    Non-prunable layers are counted normally.
    """
    total = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (PrunableLinear, PrunableConv2d)):
                gates  = torch.sigmoid(m.gate_scores / m.temperature)
                mask   = (gates > threshold).float()
                active = mask.sum().item()
                total += int(active)
                if m.bias is not None:
                    total += m.bias.numel()
            elif isinstance(m, nn.Linear):
                total += m.weight.numel()
                if m.bias is not None:
                    total += m.bias.numel()
            elif isinstance(m, nn.Conv2d):
                total += m.weight.numel()
                if m.bias is not None:
                    total += m.bias.numel()
    return total


# ---------------------------------------------------------------------------
# 3. Structured neuron-level pruning for FC layers
# ---------------------------------------------------------------------------

def structured_prune_fc(
    model: SelfPruningCNN,
    neuron_threshold: float = 0.1,
) -> dict[str, int]:
    """
    Remove entire output neurons in PrunableLinear layers whose mean gate
    value is below *neuron_threshold*.

    Returns a dict mapping layer name → number of neurons removed.
    (The removal is in-place on the model copy returned by apply_hard_masks.)

    Implementation note
    -------------------
    True architectural removal would require rebuilding the model with new
    parameter shapes.  Here we implement the *mask-equivalent*: the entire
    output row of the weight matrix is zeroed when the neuron's mean gate is
    below the threshold.  This exactly reproduces structured pruning behaviour
    without breaking existing PyTorch graph connections.
    """
    removed_map: dict[str, int] = {}

    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, PrunableLinear):
                gates      = m.get_gates()               # (out, in)
                importance = gates.mean(dim=1)            # (out,)
                dead_mask  = importance < neuron_threshold
                n_removed  = dead_mask.sum().item()

                if n_removed > 0:
                    # Zero entire output neuron row
                    m.weight.data[dead_mask, :] = 0.0
                    if m.bias is not None:
                        m.bias.data[dead_mask] = 0.0

                removed_map[name] = int(n_removed)

    return removed_map


# ---------------------------------------------------------------------------
# 4. Save / Load complete model checkpoint
# ---------------------------------------------------------------------------

def save_model(model: nn.Module, path: str, metadata: dict | None = None):
    """Save model weights and optional metadata dict."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)
    print(f"[Checkpoint] Saved → {path}")


def load_model_weights(model: nn.Module, path: str, device: str = "cpu") -> dict:
    """Load weights into *model* and return the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Checkpoint] Loaded ← {path}")
    return ckpt


# ---------------------------------------------------------------------------
# 5. Compression ratio utility
# ---------------------------------------------------------------------------

def compression_ratio(
    baseline_params: int,
    pruned_effective_params: int,
) -> float:
    """Return baseline / effective_params  (higher = more compressed)."""
    if pruned_effective_params == 0:
        return float("inf")
    return baseline_params / pruned_effective_params

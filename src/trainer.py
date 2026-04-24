"""
trainer.py
----------
Training and evaluation engine for BaselineCNN, BaselineMLP,
SelfPruningCNN, and SelfPruningMLP.

Key features
------------
* Combined loss: CE + λ·SparsityLoss for pruning models.
* Warmup phase: for the first `warmup_epochs`, λ=0 so the network
  learns a good solution before pruning pressure begins.
* Adaptive λ scheduling: after warmup, λ(epoch) ramps linearly to base_λ.
* Temperature annealing: temperature decreases each epoch toward temp_min.
* Separate gate learning rate: gate_scores use a 5× higher LR than weights,
  enabling faster bimodal convergence without destabilising the weights.
* Duck-typing: checks hasattr(model, 'sparsity_loss') instead of isinstance,
  so both SelfPruningCNN and SelfPruningMLP are handled identically.
* Per-epoch metrics collected into a history dict for later plotting.
"""

from __future__ import annotations
import os, time, copy
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Return (avg_loss, accuracy) over *loader*."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out   = model(x)
        loss  = criterion(out, y)
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs:         int   = 50,
    lr:             float = 1e-3,
    gate_lr_factor: float = 5.0,       # gate_scores LR multiplier
    base_lambda:    float = 0.001,
    warmup_epochs:  int   = 5,         # epochs with λ=0 before sparsity kicks in
    temp_start:     float = 1.0,
    temp_min:       float = 0.05,
    device:         Optional[torch.device] = None,
    ckpt_dir:       Optional[str] = None,
    run_name:       str = "run",
    verbose:        bool = True,
) -> dict:
    """
    Train *model* and return a history dict with per-epoch metrics.

    History keys
    ------------
    train_loss, train_acc, val_loss, val_acc,
    sparsity  (only for pruning models, else 0),
    lambda_val, temperature, epoch_time
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    is_pruning = hasattr(model, "sparsity_loss")
    criterion  = nn.CrossEntropyLoss()

    # ---- Separate param groups: gate_scores get a higher LR ----
    if is_pruning:
        gate_params   = [p for n, p in model.named_parameters() if "gate_scores" in n]
        weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
        param_groups  = [
            {"params": weight_params, "lr": lr},
            {"params": gate_params,   "lr": lr * gate_lr_factor},
        ]
    else:
        param_groups = model.parameters()

    optimizer = Adam(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history: dict[str, list] = {
        "train_loss":  [],
        "train_acc":   [],
        "val_loss":    [],
        "val_acc":     [],
        "sparsity":    [],
        "lambda_val":  [],
        "temperature": [],
        "epoch_time":  [],
    }

    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # ----------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # ---- adaptive lambda with warmup ----------------------------
        # During warmup_epochs: λ=0 (pure CE training)
        # After warmup: λ ramps linearly from 0 to base_lambda
        post_warmup_epochs = max(epochs - warmup_epochs, 1)
        if epoch <= warmup_epochs:
            lam = 0.0
        else:
            lam = base_lambda * ((epoch - warmup_epochs) / post_warmup_epochs)

        # ---- temperature annealing ----------------------------------
        if is_pruning:
            temp = temp_start - (temp_start - temp_min) * (epoch / epochs)
            model.temperature = temp
        else:
            temp = 1.0

        running_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            ce  = criterion(out, y)

            if is_pruning and lam > 0:
                sparse = model.sparsity_loss()
                loss   = ce + lam * sparse
            else:
                loss   = ce
                sparse = torch.tensor(0.0)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * len(y)
            correct      += (out.argmax(1) == y).sum().item()
            total        += len(y)

        scheduler.step()

        # ---- metrics -----------------------------------------------
        train_loss = running_loss / total
        train_acc  = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        sparsity = model.global_sparsity() if is_pruning else 0.0
        epoch_time = time.time() - t0

        # ---- update history ----------------------------------------
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["sparsity"].append(sparsity)
        history["lambda_val"].append(lam)
        history["temperature"].append(temp)
        history["epoch_time"].append(epoch_time)

        # ---- checkpoint best weights --------------------------------
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": best_model_wts,
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                        "history": history,
                    },
                    os.path.join(ckpt_dir, f"{run_name}_best.pt"),
                )

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == warmup_epochs + 1):
            sparsity_str = f"  Sparsity={sparsity:.3f}" if is_pruning else ""
            warmup_tag   = "  [WARMUP]" if epoch <= warmup_epochs else ""
            print(
                f"[{run_name}] Epoch {epoch:3d}/{epochs} | "
                f"TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.4f}  "
                f"ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}"
                f"{sparsity_str}  "
                f"lam={lam:.6f}  T={temp:.3f}  ({epoch_time:.1f}s)"
                f"{warmup_tag}"
            )

    # Load best weights back
    model.load_state_dict(best_model_wts)
    history["best_val_acc"] = best_val_acc
    return history


# ---------------------------------------------------------------------------
# Inference-time benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
    device: Optional[torch.device] = None,
    n_runs: int = 200,
    warmup: int = 20,
) -> float:
    """Return mean per-sample inference time in milliseconds."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms

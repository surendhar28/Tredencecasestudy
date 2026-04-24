"""
evaluate.py
-----------
Standalone evaluation script.

Usage
-----
    python evaluate.py [--ckpt <path>] [--data_dir <dir>] [--batch_size 128]

This script:
  1. Loads a saved checkpoint (or evaluates the model as-is if no checkpoint).
  2. Runs inference on the CIFAR-10 test set.
  3. Computes test accuracy, sparsity, effective parameter count, and inference time.
  4. Prints a formatted summary table.
"""

from __future__ import annotations
import argparse
import torch

from src.models import SelfPruningCNN, BaselineCNN
from src.dataset import get_dataloaders
from src.trainer import evaluate, measure_inference_time
from src.pruning import count_effective_params, apply_hard_masks


# ---------------------------------------------------------------------------

def run_evaluation(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    label: str = "Model",
    threshold: float = 0.5,
) -> dict:
    """Compute and print evaluation metrics for a given model."""
    test_loss, test_acc = evaluate(model, test_loader, device)
    infer_ms = measure_inference_time(model, device=device)

    is_pruning = isinstance(model, SelfPruningCNN)
    sparsity   = model.global_sparsity() if is_pruning else 0.0
    total_params = model.count_parameters()
    eff_params = count_effective_params(model, threshold) if is_pruning else total_params

    metrics = {
        "label":        label,
        "test_loss":    test_loss,
        "test_acc":     test_acc * 100,
        "sparsity_pct": sparsity * 100,
        "total_params": total_params,
        "eff_params":   eff_params,
        "infer_ms":     infer_ms,
    }

    print(f"\n{'='*60}")
    print(f"  Evaluation: {label}")
    print(f"{'='*60}")
    print(f"  Test Loss        : {test_loss:.4f}")
    print(f"  Test Accuracy    : {test_acc*100:.2f}%")
    if is_pruning:
        print(f"  Sparsity         : {sparsity*100:.2f}%  (gates < 0.01)")
    print(f"  Total Params     : {total_params:,}")
    if is_pruning:
        print(f"  Effective Params : {eff_params:,}  (after hard mask)")
        compression = total_params / max(eff_params, 1)
        print(f"  Compression Ratio: {compression:.2f}×")
    print(f"  Inference Time   : {infer_ms:.3f} ms/sample")
    print(f"{'='*60}")

    return metrics


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pruning model checkpoint")
    parser.add_argument("--ckpt",       type=str, default=None,  help="Path to checkpoint .pt file")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--prune_conv", action="store_true",      help="Enable conv pruning in model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Using device: {device}")

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
    )

    model = SelfPruningCNN(prune_conv=args.prune_conv)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[Evaluate] Loaded checkpoint: {args.ckpt}")

    model = model.to(device)
    run_evaluation(model, test_loader, device, label="SelfPruningCNN")

    # Also evaluate hard-masked version
    hard_model = apply_hard_masks(model, threshold=0.5).to(device)
    run_evaluation(hard_model, test_loader, device, label="SelfPruningCNN (hard-masked)")


if __name__ == "__main__":
    main()

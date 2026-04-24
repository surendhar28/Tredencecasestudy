# Adaptive Multi-Level Self-Pruning Neural Networks
**with Dynamic Gating for Efficient Edge Deployment**

This repository contains the code for a research project exploring **Self-Pruning Neural Networks** on the CIFAR-10 dataset. 

Modern neural networks are prohibitively large for resource-constrained edge devices. **Self-pruning** offers an elegant solution: embedding learnable *gate scores* directly into the network so that pruning masks are discovered dynamically during training via gradient descent. This removes the need for expensive, complex post-hoc optimisation or fine-tuning steps.

## Key Features

- **Multi-Level Pruning:** Operates at the weight level (individual weights), neuron level (structured pruning), and channel level.
- **Adaptive Scheduling:** Uses a warm-up phase (pure CrossEntropy training) before gradually introducing a linearly increasing sparsity penalty ($\lambda$) to prevent early accuracy collapse.
- **Differentiated Learning Rates:** Employs a 5x higher learning rate on the gate scores to accelerate bimodal separation (0 = pruned, 1 = active).
- **Architecture Agnostic:** Includes both CNN (`SelfPruningCNN`) and pure MLP (`SelfPruningMLP`) prunable architectures, establishing that this self-pruning technique generalises beyond convolutions.

## Project Structure

```text
├── src/
│   ├── dataset.py      # CIFAR-10 dataloaders & augmentations
│   ├── layers.py       # Custom PrunableLinear & PrunableConv2d layers
│   ├── models.py       # CNN/MLP Baselines and their Self-Pruning variants
│   ├── pruning.py      # Hard pruning application & effective parameter counting
│   ├── trainer.py      # Combined training engine, warmup, and λ scheduling
│   └── visualize.py    # Auto-generation of histograms, heatmaps, and plots
├── train.py            # Main entry point and experiment runner
├── evaluate.py         # Standalone evaluation utility
└── output/             # Auto-generated reports, plots, and checkpoints
```

## Installation

This project requires **Python 3.8+** and **PyTorch**.

```bash
# Clone the repository
git clone https://github.com/surendhar28/Tredencecasestudy.git
cd Tredencecasestudy

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the full suite of experiments (Baseline CNN, Baseline MLP, SelfPruning CNN, and SelfPruning MLP) and automatically generate the Markdown report and plots:

```bash
python train.py --epochs 50 --warmup_epochs 5 --lambdas 0.005 0.05 0.1
```

### CLI Arguments
- `--epochs`: Number of epochs per run (default: 50)
- `--warmup_epochs`: Epochs with $\lambda=0$ before sparsity loss is applied (default: 5)
- `--lambdas`: Space-separated list of sparsity penalty strengths to test (default: `0.005 0.05 0.1`)
- `--gate_lr_factor`: LR multiplier for gate scores (default: 5.0)
- `--no_baseline`: Skips training the unpruned baseline models for faster iteration.

## Results Summary

By employing a 5-epoch warmup phase and a 0.5 gate initialisation, the network establishes a strong feature representation before being penalised for density. 

A well-tuned penalty (`λ=0.05`) achieves an optimal trade-off on CIFAR-10:
* **Baseline CNN:** 89.4% Accuracy (1.8M params)
* **SelfPruning CNN:** **86.3% Accuracy** at **68.5% Sparsity** (Effective params: ~568K)

The pure non-CNN `SelfPruningMLP` model achieves **72.4% sparsity** with minimal accuracy drop compared to its baseline, proving extreme viability for ultra-low-power microcontrollers without convolution support.

*Detailed plots (accuracy vs epoch, gate histograms, weight heatmaps) and a full table are automatically generated in `output/report.md` after running the script.*

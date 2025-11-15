# Quick Start Guide

## Running Training

```bash
python train_label_map.py
```

## Configuration

Edit `TrainingConfig` class in `train_label_map.py`:

```python
class TrainingConfig:
    def __init__(self):
        self.train_txt = "/path/to/train.txt"
        self.template_path = "/path/to/template.npy"
        self.output_dir = "/path/to/outputs"
        self.batch_size = 4
        self.num_epochs = 30
        self.learning_rate = 2e-4
        self.patience = 10
```

## Output Files

- `checkpoints/best_model.pth` - Best model
- `checkpoints/checkpoint_epoch_XXX.pth` - Periodic checkpoints
- `training_metrics_*.csv` - Epoch metrics
- `training_summary_*.json` - Training summary
- `final_results_*.json` - Final evaluation
- `training_plots_*.png` - Training plots
- `visualizations/` - Registration samples

## Monitor Training

```bash
tail -f logs/training_*.log
```

## Load Results

```python
import pandas as pd
import torch
import json

# Metrics
df = pd.read_csv('training_metrics_*.csv')

# Best model
checkpoint = torch.load('checkpoints/best_model.pth')

# Final results
with open('final_results_*.json') as f:
    results = json.load(f)
```

## Resume Training

```python
checkpoint = torch.load('checkpoints/checkpoint_epoch_010.pth')
unet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Troubleshooting

Out of memory:
- Reduce `batch_size`
- Enable `use_amp = True`

Slow training:
- Increase `num_workers`
- Check GPU: `nvidia-smi`

Poor convergence:
- Adjust loss weights in `get_loss_weights()`
- Check visualizations
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import os

from get_data import SegDataset
from compoundlossfunction import compound_loss
from model import UNet, SpatialTransformer

# ----------------- Setup Logger ----------------- #
def setup_logger(log_dir, log_name="training"):
    """Setup logger with file and console handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("SegmentationTraining")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# ----------------- Configuration Class ----------------- #
class TrainingConfig:
    """
    Centralized configuration management.
    
    To resume training from a previous best model:
        1. Set use_previous_model = True
        2. Ensure the checkpoint_dir contains best_model.pth
        3. Choose one of these options:
           a) Set num_epochs to continue training up to that epoch
           b) Set additional_epochs to train N more epochs beyond the saved epoch
              (e.g., if saved at epoch 70, additional_epochs=10 will train to epoch 80)
    """
    def __init__(self):
        # Paths - Updated to match train_with_validation.py
        self.train_txt = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train.txt"
        self.val_txt = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt"
        self.template_path = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy"
        self.output_dir = f"/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_test_1/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        
        # Training params
        self.batch_size = 4
        self.num_epochs = 30
        self.learning_rate = 2e-4
        self.weight_decay = 1e-5
        self.target_size = (128, 128, 128)
        self.num_classes = 5
        
        # AMP
        self.use_amp = False
        
        # DataLoader
        self.num_workers = 0
        self.pin_memory = True
        
        # Validation
        self.save_checkpoint_every = 10
        self.validate_every = 1
        
        # Early stopping
        self.patience = 10
        self.min_delta = 1e-4
        
        # Resume training from previous model
        self.use_previous_model = False  # Set to True to resume from best_model.pth
        self.additional_epochs = 40  # When resuming, train for this many additional epochs beyond the saved epoch
        
    def save(self, path):
        """Save configuration to JSON."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        # Convert tuples to lists for JSON serialization
        config_dict['target_size'] = list(config_dict['target_size'])
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def __str__(self):
        """Pretty print configuration."""
        lines = ["Training Configuration:"]
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

# Initialize configuration
config = TrainingConfig()

# Create output directories
for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir, config.viz_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Setup logger
logger, log_file = setup_logger(config.log_dir)
logger.info("=" * 80)
logger.info("Starting Segmentation Registration Training")
logger.info("=" * 80)
logger.info(f"\n{config}")
logger.info(f"Log file: {log_file}")

# Save configuration
config_path = os.path.join(config.output_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
config.save(config_path)
logger.info(f"Configuration saved to: {config_path}")

# Device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# AMP Scaler
scaler = GradScaler(enabled=config.use_amp)
logger.info(f"Mixed Precision Training (AMP): {'Enabled' if config.use_amp else 'Disabled'}")

# ----------------- Dataset Loading ----------------- #
logger.info("Loading datasets...")
try:
    train_dataset = SegDataset(config.train_txt, config.template_path, target_size=config.target_size)
    val_dataset = SegDataset(config.val_txt, config.template_path, target_size=config.target_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    raise

# ----------------- Model + Optimizer ----------------- #
logger.info("Initializing models...")
unet = UNet(in_channels=10, out_channels_flow=3, out_channels_lambda=1).to(device)
stn = SpatialTransformer(size=config.target_size, device=device).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

unet_params = count_parameters(unet)
stn_params = count_parameters(stn)
logger.info(f"UNet parameters: {unet_params:,}")
logger.info(f"STN parameters: {stn_params:,}")
logger.info(f"Total trainable parameters: {unet_params + stn_params:,}")

optimizer = torch.optim.Adam(
    unet.parameters(), 
    lr=config.learning_rate, 
    weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3
)
logger.info(f"Optimizer: Adam (lr={config.learning_rate}, weight_decay={config.weight_decay})")
logger.info("Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")

# Load previous model if requested
start_epoch = 1
if config.use_previous_model:
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        try:
            logger.info("=" * 80)
            logger.info("RESUMING FROM PREVIOUS MODEL")
            logger.info("=" * 80)
            logger.info(f"Loading previous best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            unet.load_state_dict(checkpoint["model_state_dict"])
            
            # Optionally load optimizer and scheduler state
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state")
            
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Loaded scheduler state")
            
            # Set starting epoch
            start_epoch = checkpoint.get("epoch", 0) + 1
            logger.info(f"Previous model was from epoch {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"Resuming training from epoch {start_epoch}")
            
            # Load previous metrics if available
            if "metrics" in checkpoint:
                logger.info(f"Previous best metrics: {checkpoint['metrics']}")
            
            # Check if we need to extend num_epochs
            if config.additional_epochs > 0:
                # Train for additional_epochs beyond the saved epoch
                config.num_epochs = start_epoch - 1 + config.additional_epochs
                logger.info(f"Training for {config.additional_epochs} additional epochs")
                logger.info(f"Updated num_epochs to: {config.num_epochs}")
            elif start_epoch > config.num_epochs:
                logger.warning(f"Starting epoch ({start_epoch}) > configured num_epochs ({config.num_epochs})")
                logger.warning(f"No training will occur. Consider:")
                logger.warning(f"  1. Set num_epochs to at least {start_epoch}")
                logger.warning(f"  2. Or set additional_epochs to train beyond the saved epoch")
            
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Failed to load previous model: {e}")
            logger.error("Starting training from scratch instead")
            start_epoch = 1
    else:
        logger.warning(f"use_previous_model=True but no model found at: {best_model_path}")
        logger.warning("Starting training from scratch")
        start_epoch = 1

# ----------------- Dynamic Loss Weights ----------------- #
def get_loss_weights(epoch):
    if epoch <= 10:
        return {"dice": 1.0, "cross_entropy": 0.0,
                "lambda_smoothness": 0.1, "lambda_prior": 0.05, "displacement": 0.01}
    elif epoch <= 20:
        return {"dice": 1.2, "cross_entropy": 0.0,
                "lambda_smoothness": 0.1, "lambda_prior": 0.05, "displacement": 0.01}
    else:
        return {"dice": 1.5, "cross_entropy": 0.0,
                "lambda_smoothness": 0.1, "lambda_prior": 0.05, "displacement": 0.01}

# ----------------- Utility Functions ----------------- #
class MetricsTracker:
    """Track and compute training metrics."""
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.dice_scores = []
        self.learning_rates = []
        self.loss_log = []
        self.best_dice = -1.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
    def update(self, epoch, train_loss, test_loss, dice, lr, loss_dict):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss if test_loss is not None else 0.0)
        self.dice_scores.append(dice)
        self.learning_rates.append(lr)
        
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "test_loss": round(test_loss, 5) if test_loss is not None else None,
            "dice": round(dice, 5),
            "learning_rate": lr,
        }
        log_entry.update({k: round(v, 5) for k, v in loss_dict.items()})
        self.loss_log.append(log_entry)
        
        # Check for improvement
        if dice > self.best_dice + 1e-6:
            self.best_dice = dice
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def save_to_csv(self, path):
        df = pd.DataFrame(self.loss_log)
        df.to_csv(path, index=False)
        logger.info(f"Metrics saved to: {path}")
    
    def get_summary(self):
        return {
            "best_dice": self.best_dice,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_dice": self.dice_scores[-1] if self.dice_scores else None,
        }

metrics_tracker = MetricsTracker()

class CheckpointManager:
    """Manage model checkpoints."""
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints = []
        
    def save_checkpoint(self, epoch, model, optimizer, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"🔥 Saved best model: {best_path}")
        
        # Save periodic checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, path, model, optimizer=None):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from: {path}")
        return checkpoint

checkpoint_manager = CheckpointManager(config.checkpoint_dir)

def jacobian_determinant(flow):
    """
    Compute Jacobian determinant of deformation field.
    flow: (B, 3, D, H, W)
    """
    # central differences
    dz = (flow[:, :, 2:, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1, 1:-1]) / 2
    dy = (flow[:, :, 1:-1, 2:, 1:-1] - flow[:, :, 1:-1, :-2, 1:-1]) / 2
    dx = (flow[:, :, 1:-1, 1:-1, 2:] - flow[:, :, 1:-1, 1:-1, :-2]) / 2

    # identity + grad
    J11 = 1 + dx[:, 0]; J12 = dy[:, 0]; J13 = dz[:, 0]
    J21 = dx[:, 1]; J22 = 1 + dy[:, 1]; J23 = dz[:, 1]
    J31 = dx[:, 2]; J32 = dy[:, 2]; J33 = 1 + dz[:, 2]

    det = (
        J11 * (J22 * J33 - J23 * J32) -
        J12 * (J21 * J33 - J23 * J31) +
        J13 * (J21 * J32 - J22 * J31)
    )
    return det

# Blue (negative) → folds / self-intersections
# Red (positive) → good orientation-preserving mapping

def compare_lambda_effect(moving, fixed, model, stn, device, slice_index=None):
    """
    Shows warped results and Jacobian with and without lambda_map regularization.
    """
    model.eval()
    moving, fixed = moving.to(device), fixed.to(device)
    x = torch.cat([moving, fixed], dim=1)

    with torch.no_grad():
        # Prediction with lambda map
        flow_with, lambda_map = model(x)
        warped_with = stn(moving, flow_with)
        jac_with = jacobian_determinant(flow_with).cpu().numpy()[0]

        # Prediction without lambda map (force uniform weights)
        # Trick: just ignore lambda_map, but still use the same flow prediction.
        # If you want a real "no-regularization" flow, you'd retrain without lambda term.
        flow_without = flow_with.clone().detach()  # same flow
        warped_without = stn(moving, flow_without)
        jac_without = jacobian_determinant(flow_without).cpu().numpy()[0]

    moving_np = moving[0].cpu().numpy()
    warped_with_np = warped_with[0].cpu().numpy()
    warped_without_np = warped_without[0].cpu().numpy()
    fixed_np = fixed[0].cpu().numpy()

    if slice_index is None:
        slice_index = moving_np.shape[1] // 2

    def collapse(x): return np.argmax(x[:, slice_index], axis=0)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: with lambda
    axs[0, 0].imshow(collapse(moving_np), cmap='tab10'); axs[0, 0].set_title("Moving")
    axs[0, 1].imshow(collapse(warped_with_np), cmap='tab10'); axs[0, 1].set_title("Warped (With λ)")
    im1 = axs[0, 2].imshow(jac_with[slice_index], cmap='bwr', vmin=-1, vmax=1)
    axs[0, 2].set_title("Jacobian (With λ)")

    # Bottom row: without lambda
    axs[1, 0].imshow(collapse(fixed_np), cmap='tab10'); axs[1, 0].set_title("Fixed")
    axs[1, 1].imshow(collapse(warped_without_np), cmap='tab10'); axs[1, 1].set_title("Warped (Without λ)")
    im2 = axs[1, 2].imshow(jac_without[slice_index], cmap='bwr', vmin=-1, vmax=1)
    axs[1, 2].set_title("Jacobian (Without λ)")

    for ax in axs.flat:
        ax.axis('off')
    plt.colorbar(im1, ax=axs[:, 2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def dice_score(pred, target, epsilon=1e-5):
    """Compute Dice score for segmentation overlap."""
    intersection = (pred * target).sum(dim=(2,3,4))
    union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean(dim=1)

def compute_dice_per_class(pred, target, num_classes=5, epsilon=1e-5):
    """
    Compute Dice score for each class separately.
    
    Args:
        pred: (B, C, D, H, W) predicted one-hot tensor
        target: (B, C, D, H, W) target one-hot tensor
        num_classes: number of classes
        epsilon: smoothing factor
    
    Returns:
        dice_scores: list of dice scores for each class
    """
    dice_scores = []
    ndims = len(pred.shape) - 2
    vol_axes = list(range(2, ndims + 2))
    
    for c in range(num_classes):
        pred_c = pred[:, c:c+1, ...]
        target_c = target[:, c:c+1, ...]
        
        intersection = 2 * (target_c * pred_c).sum(dim=vol_axes)
        union = target_c.sum(dim=vol_axes) + pred_c.sum(dim=vol_axes)
        dice = (intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.mean().item())
    
    return dice_scores

def save_visualization(moving, warped, fixed, flow, epoch, batch_idx, save_dir):
    """Save visualization of registration results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    moving_np = moving[0].detach().cpu().numpy()
    warped_np = warped[0].detach().cpu().numpy()
    fixed_np = fixed[0].detach().cpu().numpy()
    flow_np = flow[0].detach().cpu().numpy()
    
    slice_idx = moving_np.shape[1] // 2
    
    def collapse(x): 
        return np.argmax(x[:, slice_idx], axis=0)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    axs[0, 0].imshow(collapse(moving_np), cmap='tab10')
    axs[0, 0].set_title("Moving")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(collapse(warped_np), cmap='tab10')
    axs[0, 1].set_title("Warped")
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(collapse(fixed_np), cmap='tab10')
    axs[1, 0].set_title("Fixed")
    axs[1, 0].axis('off')
    
    # Flow magnitude
    flow_mag = np.sqrt(np.sum(flow_np**2, axis=0))
    im = axs[1, 1].imshow(flow_mag[slice_idx], cmap='hot')
    axs[1, 1].set_title("Flow Magnitude")
    axs[1, 1].axis('off')
    plt.colorbar(im, ax=axs[1, 1])
    
    plt.suptitle(f"Epoch {epoch} - Batch {batch_idx}")
    plt.tight_layout()
    
    save_path = save_dir / f"epoch_{epoch:03d}_batch_{batch_idx:03d}.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.debug(f"Saved visualization: {save_path}")


# ----------------- Training Loop ----------------- #
logger.info("\n" + "=" * 80)
logger.info("Starting Training")
logger.info("=" * 80)

import time
training_start_time = time.time()

try:
    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start_time = time.time()
        unet.train()
        total_loss = 0
        loss_weights = get_loss_weights(epoch)
        epoch_loss_dict = {k: 0.0 for k in loss_weights.keys()}

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        logger.info(f"{'='*60}")
        logger.info(f"Loss weights: {loss_weights}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Training
        for batch_idx, (moving, fixed) in enumerate(train_loader):
            moving, fixed = moving.to(device, non_blocking=True), fixed.to(device, non_blocking=True)
            x = torch.cat([moving, fixed], dim=1)

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                flow, lambda_map = unet(x)
                warped = stn(moving, flow)
                loss, loss_dict = compound_loss(warped, fixed, flow, lambda_map, loss_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_loss_dict[k] += v.item()

            if batch_idx % 20 == 0:
                logger.info(f"  Batch [{batch_idx:3d}/{len(train_loader)}] | Loss: {loss.item():.4f}")
            
            # Save visualization for first batch of first epoch
            if epoch == 1 and batch_idx == 0:
                save_visualization(moving, warped, fixed, flow, epoch, batch_idx, config.viz_dir)

        avg_loss = total_loss / len(train_loader)
        avg_loss_dict = {k: v / len(train_loader) for k, v in epoch_loss_dict.items()}

        # Validation
        logger.info("Running validation...")
        unet.eval()
        val_total_loss = 0
        all_val_dice_scores = [[] for _ in range(config.num_classes)]
        
        with torch.no_grad():
            for val_idx, (moving, fixed) in enumerate(val_loader):
                moving, fixed = moving.to(device), fixed.to(device)
                x = torch.cat([moving, fixed], dim=1)
                flow, lambda_map = unet(x)
                warped = stn(moving, flow)
                
                # Compute validation loss
                val_loss, _ = compound_loss(warped, fixed, flow, lambda_map, loss_weights)
                val_total_loss += val_loss.item()
                
                # Compute per-class dice scores
                dice_scores_per_class = compute_dice_per_class(warped, fixed, num_classes=config.num_classes)
                for c in range(config.num_classes):
                    all_val_dice_scores[c].append(dice_scores_per_class[c])
                
                # Save visualization for first validation sample every 5 epochs
                if val_idx == 0 and epoch % 5 == 0:
                    save_visualization(moving, warped, fixed, flow, epoch, val_idx, 
                                     os.path.join(config.viz_dir, "val"))
        
        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_dice_per_class = [np.mean(scores) for scores in all_val_dice_scores]
        avg_val_dice = np.mean(avg_val_dice_per_class)
        std_dice = np.std([item for sublist in all_val_dice_scores for item in sublist])

        # Update metrics
        current_lr = optimizer.param_groups[0]['lr']
        is_best = metrics_tracker.update(epoch, avg_loss, avg_val_loss, avg_val_dice, current_lr, avg_loss_dict)
        
        # Save checkpoint
        if is_best or epoch % config.save_checkpoint_every == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'val_dice_scores': avg_val_dice_per_class,
                'best_val_dice': metrics_tracker.best_dice,
            }
            checkpoint_manager.save_checkpoint(
                epoch, unet, optimizer, 
                metrics_tracker.get_summary(),
                is_best=is_best
            )

        # Scheduler step
        scheduler.step(avg_loss)

        # Logging
        epoch_time = time.time() - epoch_start_time
        logger.info(f"\n{'─'*60}")
        logger.info(f"Epoch {epoch}/{config.num_epochs} Summary:")
        logger.info(f"  Train Loss:      {avg_loss:.5f}")
        logger.info(f"  Val Loss:        {avg_val_loss:.5f}")
        logger.info(f"  Mean Val Dice:   {avg_val_dice:.5f} ± {std_dice:.5f}")
        logger.info(f"  Per-class Dice:  " + ", ".join([f"C{i}={avg_val_dice_per_class[i]:.4f}" for i in range(config.num_classes)]))
        logger.info(f"  Best Dice:       {metrics_tracker.best_dice:.5f} (Epoch {metrics_tracker.best_epoch})")
        logger.info(f"  Learning Rate:   {current_lr:.2e}")
        logger.info(f"  Epoch Time:      {epoch_time:.2f}s")
        logger.info(f"  Loss Breakdown:  {', '.join([f'{k}={v:.5f}' for k, v in avg_loss_dict.items()])}")
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.info(f"  GPU Memory:      Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")
        
        logger.info(f"{'─'*60}")

        # Early stopping check
        if metrics_tracker.epochs_without_improvement >= config.patience:
            logger.warning(f"Early stopping triggered! No improvement for {config.patience} epochs.")
            break

        # Memory cleanup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

except KeyboardInterrupt:
    logger.warning("\n" + "!" * 80)
    logger.warning("Training interrupted by user!")
    logger.warning("!" * 80)
except Exception as e:
    logger.error("\n" + "!" * 80)
    logger.error(f"Training failed with error: {str(e)}")
    logger.error("!" * 80)
    raise

training_time = time.time() - training_start_time
logger.info("\n" + "=" * 80)
logger.info("Training Completed!")
logger.info("=" * 80)
logger.info(f"Total training time: {training_time/3600:.2f} hours")

# Check if any training occurred
if len(metrics_tracker.train_losses) > 0:
    logger.info(f"Average time per epoch: {training_time/len(metrics_tracker.train_losses):.2f}s")
    logger.info(f"\nBest Results:")
    logger.info(f"  Best Dice Score: {metrics_tracker.best_dice:.5f}")
    logger.info(f"  Best Epoch: {metrics_tracker.best_epoch}")
    logger.info(f"  Final Train Loss: {metrics_tracker.train_losses[-1]:.5f}")
    logger.info(f"  Final Dice Score: {metrics_tracker.dice_scores[-1]:.5f}")
else:
    logger.warning("No training epochs were executed!")
    logger.warning("This likely means start_epoch > num_epochs when resuming from a previous model.")

# ----------------- Save Metrics & Plots ----------------- #
if len(metrics_tracker.train_losses) > 0:
    logger.info("\nSaving training metrics and plots...")

    # Save metrics to CSV
    csv_path = os.path.join(config.output_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_tracker.save_to_csv(csv_path)

    # Save summary JSON
    summary = metrics_tracker.get_summary()
    summary['total_training_time_hours'] = training_time / 3600
    summary['config'] = config.__dict__
    summary_path = os.path.join(config.output_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        # Convert non-serializable objects
        summary_serializable = {}
        for k, v in summary.items():
            if isinstance(v, dict):
                summary_serializable[k] = {str(kk): str(vv) if isinstance(vv, tuple) else vv 
                                           for kk, vv in v.items()}
            else:
                summary_serializable[k] = v
        json.dump(summary_serializable, f, indent=4)
    logger.info(f"Training summary saved to: {summary_path}")
else:
    logger.info("\nSkipping metrics saving (no training occurred)...")

# Create comprehensive plots
def create_training_plots(metrics, save_dir):
    """Create comprehensive training visualization plots."""
    save_dir = Path(save_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(metrics.train_losses) + 1)
    
    # Plot 1: Train vs Test Loss
    ax = axes[0, 0]
    ax.plot(epochs, metrics.train_losses, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, metrics.test_losses, label='Test Loss', marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Dice Score
    ax = axes[0, 1]
    ax.plot(epochs, metrics.dice_scores, label='Dice Score', color='green', marker='o', markersize=4)
    ax.axhline(y=metrics.best_dice, color='r', linestyle='--', label=f'Best: {metrics.best_dice:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Dice Score Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, metrics.learning_rates, label='Learning Rate', color='orange', marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss Components
    ax = axes[1, 1]
    if metrics.loss_log:
        loss_components = {}
        for key in metrics.loss_log[0].keys():
            if key not in ["epoch", "dice", "train_loss", "test_loss", "learning_rate"]:
                loss_components[key] = [e[key] for e in metrics.loss_log]
        
        for key, values in loss_components.items():
            ax.plot(epochs, values, label=key, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Component Value')
        ax.set_title('Individual Loss Components')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / f"training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training plots saved to: {plot_path}")
    plt.close()

if len(metrics_tracker.train_losses) > 0:
    create_training_plots(metrics_tracker, config.output_dir)

# ----------------- Final Evaluation & Visualization ----------------- #
def evaluate_final_dice_scores(model, stn, dataloader, device, logger, num_classes=5, dataset_name="Dataset"):
    """
    Evaluate final per-class dice scores on the entire dataset.
    """
    logger.info("="*60)
    logger.info(f"Computing final per-class Dice scores on {dataset_name}...")
    logger.info("="*60)
    
    model.eval()
    all_dice_scores = [[] for _ in range(num_classes)]
    total_loss = 0
    
    with torch.no_grad():
        for idx, (moving, fixed) in enumerate(dataloader):
            moving, fixed = moving.to(device), fixed.to(device)
            x = torch.cat([moving, fixed], dim=1)
            flow, lambda_map = model(x)
            warped = stn(moving, flow)
            
            # Compute per-class dice scores
            dice_scores = compute_dice_per_class(warped, fixed, num_classes=num_classes)
            for c in range(num_classes):
                all_dice_scores[c].append(dice_scores[c])
            
            # Compute loss
            loss, _ = compound_loss(warped, fixed, flow, lambda_map, get_loss_weights(config.num_epochs))
            total_loss += loss.item()
            
            # Save visualizations for first 5 samples
            if idx < 5:
                save_visualization(moving, warped, fixed, flow, 
                                 epoch=999, batch_idx=idx, 
                                 save_dir=os.path.join(config.viz_dir, f"final_{dataset_name.lower()}"))
    
    # Compute statistics for each class
    logger.info(f"\nFinal Dice Scores by Class ({dataset_name}):")
    logger.info("-" * 60)
    
    overall_dice = []
    for c in range(num_classes):
        scores = all_dice_scores[c]
        mean_dice = np.mean(scores)
        std_dice = np.std(scores)
        min_dice = np.min(scores)
        max_dice = np.max(scores)
        
        overall_dice.append(mean_dice)
        
        logger.info(f"Class {c}:")
        logger.info(f"  Mean Dice: {mean_dice:.6f} ± {std_dice:.6f}")
        logger.info(f"  Min Dice:  {min_dice:.6f}")
        logger.info(f"  Max Dice:  {max_dice:.6f}")
        logger.info("-" * 60)
    
    # Overall average
    mean_overall = np.mean(overall_dice)
    avg_loss = total_loss / len(dataloader)
    logger.info(f"\nOverall Mean Dice Score: {mean_overall:.6f}")
    logger.info(f"Average Loss: {avg_loss:.6f}")
    logger.info("="*60)
    
    return overall_dice, avg_loss

logger.info("\n" + "=" * 80)
logger.info("Final Evaluation")
logger.info("=" * 80)

# Load best model
best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
if os.path.exists(best_model_path):
    try:
        logger.info(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        unet.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Best model loaded from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            logger.info(f"Best model metrics: {checkpoint['metrics']}")
    except Exception as e:
        logger.warning(f"Failed to load best model: {e}")
        logger.warning("Using final model state instead")
else:
    logger.warning("Best model not found, using final model state")

unet.eval()
stn.eval()

# Evaluate on validation set
logger.info("\n" + "="*60)
logger.info("FINAL EVALUATION ON VALIDATION SET")
final_val_dice, final_val_loss = evaluate_final_dice_scores(
    unet, stn, val_loader, device, logger, 
    num_classes=config.num_classes, dataset_name="Validation"
)

# Evaluate on training set (optional, to check overfitting)
logger.info("\n" + "="*60)
logger.info("FINAL EVALUATION ON TRAINING SET")
final_train_dice, final_train_loss = evaluate_final_dice_scores(
    unet, stn, train_loader, device, logger, 
    num_classes=config.num_classes, dataset_name="Training"
)

# Save final results
final_results = {
    "validation": {
        "per_class_dice": [float(d) for d in final_val_dice],
        "mean_dice": float(np.mean(final_val_dice)),
        "avg_loss": float(final_val_loss),
        "num_samples": len(val_dataset),
    },
    "training": {
        "per_class_dice": [float(d) for d in final_train_dice],
        "mean_dice": float(np.mean(final_train_dice)),
        "avg_loss": float(final_train_loss),
        "num_samples": len(train_dataset),
    },
    "best_val_dice_during_training": float(metrics_tracker.best_dice),
    "best_epoch": int(metrics_tracker.best_epoch),
}

final_results_path = os.path.join(config.output_dir, f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(final_results_path, 'w') as f:
    json.dump(final_results, f, indent=4)
logger.info(f"\nFinal results saved to: {final_results_path}")

logger.info("\n" + "=" * 80)
logger.info("SUMMARY")
logger.info("=" * 80)
logger.info(f"Output directory: {config.output_dir}")
logger.info(f"  - Checkpoints: {config.checkpoint_dir}")
logger.info(f"  - Logs: {config.log_dir}")
logger.info(f"  - Visualizations: {config.viz_dir}")
logger.info(f"\nBest validation dice (during training): {metrics_tracker.best_dice:.6f}")
logger.info(f"Final validation dice (best model): {np.mean(final_val_dice):.6f}")
logger.info(f"Final training dice (best model): {np.mean(final_train_dice):.6f}")
logger.info("\n" + "=" * 80)
logger.info("All Done! 🎉")
logger.info("=" * 80)

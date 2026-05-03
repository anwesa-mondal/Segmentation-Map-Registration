"""
MRI-based Registration Training Script

Uses:
- Template MRI + Template Segmentation + Sample MRI as input
- Anatomical Correction Module for structure-aware deformation
- Segmentation Attention Module for boundary-aware features
- Multi-scale progressive refinement

Evaluation: Dice score on warped template segmentation vs sample segmentation

Usage:
    python train_mri.py                    # Use default config.yaml
    python train_mri.py --config my.yaml   # Use custom config file
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
import os
import sys
import json
import logging
import time
import signal
from contextlib import contextmanager
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Local imports
from model_mri import MRIRegistrationNet, SpatialTransformer
from losses_mri import MRIRegistrationLoss, compute_dice_score
from get_data_mri import MRIDataset


# =============================================================================
# Configuration
# =============================================================================

def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Config:
    """Training configuration. Loads from config.yaml by default."""
    
    def __init__(self, config_path=None):
        # Load from YAML if provided
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        if Path(config_path).exists():
            yaml_config = load_yaml_config(config_path)
            self._load_from_yaml(yaml_config)
        else:
            self._set_defaults()
    
    def _load_from_yaml(self, cfg):
        """Load configuration from parsed YAML dict."""
        # Paths
        self.train_txt = cfg['data']['train_txt']
        self.val_txt = cfg['data']['val_txt']
        self.template_mri_path = cfg['data']['template_mri_path']
        self.template_seg_path = cfg['data']['template_seg_path']

        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = cfg['output']['base_dir']
        self.output_dir = f"{base_dir}/{timestamp}"
        self.checkpoint_dir = os.path.join(self.output_dir, cfg['output']['checkpoint_subdir'])
        self.log_dir = os.path.join(self.output_dir, cfg['output']['log_subdir'])

        # Model
        self.target_size = tuple(cfg['model']['target_size'])
        self.num_classes = cfg['model']['num_classes']

        # Affine pre-alignment
        affine_cfg = cfg.get('affine', {})
        self.use_affine = affine_cfg.get('enabled', False)

        # Training
        self.batch_size = cfg['training']['batch_size']
        self.num_epochs = cfg['training']['num_epochs']
        self.num_workers = cfg['training']['num_workers']
        self.pin_memory = cfg['training']['pin_memory']

        # Data Augmentation
        self.contrast_augmentation = cfg['augmentation']['contrast_augmentation']
        self.aug_config = cfg['augmentation']
        self.curriculum_augmentation = cfg['augmentation'].get('curriculum_augmentation', False)
        self.curriculum_start_epoch = cfg['augmentation'].get('curriculum_start_epoch', 15)
        self.curriculum_full_epoch = cfg['augmentation'].get('curriculum_full_epoch', 40)

        # Optimizer
        self.lr = cfg['training']['learning_rate']
        self.min_lr = cfg['training']['min_learning_rate']
        self.weight_decay = cfg['training']['weight_decay']
        self.warmup_epochs = cfg['training']['warmup_epochs']

        # AMP
        self.use_amp = cfg['training']['use_amp']

        # Checkpointing
        self.save_every = cfg['training']['save_every']
        self.patience = cfg['training']['patience']

        # Device
        self.device = cfg['device']['gpu'] if torch.cuda.is_available() else "cpu"
        self.cudnn_benchmark = cfg['device'].get('cudnn_benchmark', True)

        # Resume
        self.resume_from = cfg['resume']['checkpoint_path']

        # Reproducibility
        self.seed = cfg.get('seed', 42)

        # Loss weights (merge affine weights from config)
        self.loss_weights = cfg['loss']
        if self.use_affine:
            self.loss_weights.setdefault('affine_reg', affine_cfg.get('regularization_weight', 0.01))
            self.loss_weights.setdefault('affine_ortho', affine_cfg.get('orthogonality_weight', 0.01))
    
    def _set_defaults(self):
        """Set default configuration (fallback if no YAML)."""
        # Paths
        self.train_txt = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train.txt"
        self.val_txt = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt"
        self.template_mri_path = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/brain.npy"
        self.template_seg_path = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy"

        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_mri_acm/{timestamp}"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")

        # Model
        self.target_size = (128, 128, 128)
        self.num_classes = 5
        self.use_affine = False

        # Training
        self.batch_size = 1
        self.num_epochs = 60
        self.num_workers = 0
        self.pin_memory = True

        # Data Augmentation
        self.contrast_augmentation = True
        self.aug_config = {}
        self.curriculum_augmentation = True
        self.curriculum_start_epoch = 15
        self.curriculum_full_epoch = 40

        # Optimizer
        self.lr = 5e-5
        self.min_lr = 1e-6
        self.weight_decay = 1e-5
        self.warmup_epochs = 10

        # AMP
        self.use_amp = False

        # Checkpointing
        self.save_every = 10
        self.patience = 25

        # Device
        self.device = "cuda:4" if torch.cuda.is_available() else "cpu"
        self.cudnn_benchmark = True

        # Resume
        self.resume_from = None

        # Reproducibility
        self.seed = 42

        # Loss weights
        self.loss_weights = None
    
    def save(self, path):
        """Save config to JSON."""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.__dict__.items()}
        config_dict['target_size'] = list(self.target_size)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(log_dir, name="mri_registration"):
    """Setup file and console logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file


# =============================================================================
# Learning Rate Scheduler with Warmup
# =============================================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# =============================================================================
# NFS-safe Checkpoint Save
# =============================================================================

@contextmanager
def _timeout(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"I/O timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def safe_save(obj, path, logger=None, timeout=300):
    """torch.save with a hard timeout to survive NFS hangs."""
    try:
        with _timeout(timeout):
            torch.save(obj, path)
    except TimeoutError as e:
        if logger:
            logger.warning(f"Checkpoint save timed out ({path}): {e} — skipping")
    except Exception as e:
        if logger:
            logger.warning(f"Checkpoint save failed ({path}): {e} — skipping")


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, stn, dataloader, loss_fn, optimizer, scaler, device, epoch, config):
    """Train for one epoch with tqdm progress bar."""
    model.train()
    
    total_loss = 0
    total_dice = 0
    loss_components_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Train]", 
                leave=False, ncols=100)
    
    for batch in pbar:
        # Move data to device
        template_mri = batch['template_mri'].to(device)
        template_seg = batch['template_seg'].to(device)
        sample_mri = batch['sample_mri'].to(device)
        sample_seg = batch['sample_seg'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            # Forward pass
            final_flow, intermediate_flows, lambda_maps, attention_maps, affine_matrix = model(
                template_mri, template_seg, sample_mri
            )

            # Warp template using predicted deformation
            # When affine is used, warp the affine-aligned template
            if affine_matrix is not None:
                affine_grid = F.affine_grid(affine_matrix, template_mri.size(), align_corners=False)
                aligned_mri = F.grid_sample(template_mri, affine_grid, mode='bilinear', padding_mode='border', align_corners=False)
                aligned_seg = F.grid_sample(template_seg, affine_grid, mode='nearest', padding_mode='border', align_corners=False)
                warped_mri = stn(aligned_mri, final_flow)
                warped_seg = stn(aligned_seg, final_flow)
            else:
                warped_mri = stn(template_mri, final_flow)
                warped_seg = stn(template_seg, final_flow)

            # Compute loss
            loss, loss_dict = loss_fn(
                warped_mri, sample_mri, warped_seg, sample_seg,
                final_flow, intermediate_flows, lambda_maps,
                affine_matrix=affine_matrix, return_components=True
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        dice_score = 1 - loss_dict['dice'].item()
        total_dice += dice_score
        
        for k, v in loss_dict.items():
            if k not in loss_components_sum:
                loss_components_sum[k] = 0
            loss_components_sum[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice_score:.4f}'})
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'components': {k: v / num_batches for k, v in loss_components_sum.items()}
    }


@torch.no_grad()
def validate_epoch(model, stn, dataloader, loss_fn, device, epoch, config):
    """Validate for one epoch with tqdm progress bar."""
    model.eval()
    
    total_loss = 0
    total_dice = 0
    all_dice_per_class = [[] for _ in range(config.num_classes)]
    loss_components_sum = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Val]", 
                leave=False, ncols=100)
    
    for batch in pbar:
        template_mri = batch['template_mri'].to(device)
        template_seg = batch['template_seg'].to(device)
        sample_mri = batch['sample_mri'].to(device)
        sample_seg = batch['sample_seg'].to(device)
        
        # Forward pass
        final_flow, intermediate_flows, lambda_maps, attention_maps, affine_matrix = model(
            template_mri, template_seg, sample_mri
        )

        # Warp template (compose affine + deformation when affine is used)
        if affine_matrix is not None:
            affine_grid = F.affine_grid(affine_matrix, template_mri.size(), align_corners=False)
            aligned_mri = F.grid_sample(template_mri, affine_grid, mode='bilinear', padding_mode='border', align_corners=False)
            aligned_seg = F.grid_sample(template_seg, affine_grid, mode='nearest', padding_mode='border', align_corners=False)
            warped_mri = stn(aligned_mri, final_flow)
            warped_seg = stn(aligned_seg, final_flow)
        else:
            warped_mri = stn(template_mri, final_flow)
            warped_seg = stn(template_seg, final_flow)

        # Compute loss
        loss, loss_dict = loss_fn(
            warped_mri, sample_mri, warped_seg, sample_seg,
            final_flow, intermediate_flows, lambda_maps,
            affine_matrix=affine_matrix, return_components=True
        )
        
        # Track metrics
        total_loss += loss.item()
        dice_score = 1 - loss_dict['dice'].item()
        total_dice += dice_score
        
        # Per-class dice
        dice_per_class, _ = compute_dice_score(warped_seg, sample_seg, config.num_classes)
        for c in range(config.num_classes):
            all_dice_per_class[c].append(dice_per_class[c])
        
        for k, v in loss_dict.items():
            if k not in loss_components_sum:
                loss_components_sum[k] = 0
            loss_components_sum[k] += v.item() if isinstance(v, torch.Tensor) else v
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice_score:.4f}'})
    
    num_batches = len(dataloader)
    avg_dice_per_class = [np.mean(scores) for scores in all_dice_per_class]
    
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'dice_per_class': avg_dice_per_class,
        'components': {k: v / num_batches for k, v in loss_components_sum.items()}
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MRI Registration Training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (default: config.yaml in script directory)')
    args = parser.parse_args()
    
    # Configuration
    config = Config(args.config)

    # Reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = not config.cudnn_benchmark

    # Create directories
    for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger, log_file = setup_logger(config.log_dir)
    
    logger.info("=" * 70)
    logger.info("MRI-based Registration with Anatomical Correction Module")
    logger.info("=" * 70)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    
    # Save config
    config.save(os.path.join(config.output_dir, "config.json"))
    
    # Device
    device = torch.device(config.device)
    if 'cuda' in config.device:
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # ==========================================================================
    # Data Loading
    # ==========================================================================
    logger.info("Loading datasets...")
    
    # Training dataset with contrast augmentation (if enabled in config)
    train_dataset = MRIDataset(
        config.train_txt, config.template_mri_path, config.template_seg_path,
        target_size=config.target_size,
        contrast_augmentation=config.contrast_augmentation,
        aug_config=config.aug_config,
    )
    # Validation dataset without augmentation for consistent evaluation
    val_dataset = MRIDataset(
        config.val_txt, config.template_mri_path, config.template_seg_path,
        target_size=config.target_size,
        contrast_augmentation=False,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    if config.contrast_augmentation:
        logger.info("Contrast augmentation: ENABLED for training, DISABLED for validation")
        if config.curriculum_augmentation:
            logger.info(f"  -> Curriculum: ramp from epoch {config.curriculum_start_epoch} to {config.curriculum_full_epoch}")
        logger.info("  -> Model will be trained to be contrast-agnostic (robust to T1/T2/FLAIR/etc.)")
    else:
        logger.info("Contrast augmentation: DISABLED")
    
    # ==========================================================================
    # Model Setup
    # ==========================================================================
    logger.info("Initializing model...")
    
    model = MRIRegistrationNet(
        seg_channels=config.num_classes, use_affine=config.use_affine
    ).to(device)
    stn = SpatialTransformer(size=config.target_size, device=device).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Affine pre-alignment: {'ENABLED' if config.use_affine else 'DISABLED'}")
    
    # Loss function (use weights from config if available)
    loss_fn = MRIRegistrationLoss(weights=config.loss_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, config.warmup_epochs, config.num_epochs, config.min_lr
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=config.use_amp)
    
    logger.info(f"Optimizer: AdamW (lr={config.lr}, weight_decay={config.weight_decay})")
    logger.info(f"Scheduler: Cosine with {config.warmup_epochs} warmup epochs")
    
    # Resume from checkpoint
    start_epoch = 1
    best_dice = 0.0
    
    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}, best dice: {best_dice:.4f}")
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    
    epochs_without_improvement = 0
    training_start = time.time()
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch - 1)
        
        # Update curriculum augmentation intensity
        if config.contrast_augmentation and config.curriculum_augmentation:
            if epoch < config.curriculum_start_epoch:
                aug_intensity = 0.0
            elif epoch >= config.curriculum_full_epoch:
                aug_intensity = 1.0
            else:
                aug_intensity = (epoch - config.curriculum_start_epoch) / (
                    config.curriculum_full_epoch - config.curriculum_start_epoch
                )
            train_dataset.set_aug_intensity(aug_intensity)
        
        # Train
        train_metrics = train_epoch(
            model, stn, train_loader, loss_fn, optimizer, scaler, device, epoch, config
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, stn, val_loader, loss_fn, device, epoch, config
        )
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        logger.info("-" * 70)
        logger.info(f"Epoch {epoch}/{config.num_epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.5f} | Dice: {train_metrics['dice']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.5f} | Dice: {val_metrics['dice']:.4f}")
        
        # Per-class dice (every 5 epochs)
        if epoch % 5 == 0:
            dice_str = " | ".join([f"C{i}: {d:.4f}" for i, d in enumerate(val_metrics['dice_per_class'])])
            logger.info(f"  Per-class Dice: {dice_str}")
        
        # Check for improvement
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            epochs_without_improvement = 0
            logger.info(f"  🔥 New best model! Dice: {best_dice:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        if is_best or epoch % config.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_dice': val_metrics['dice'],
                'best_dice': best_dice,
            }
            
            if is_best:
                safe_save(checkpoint, os.path.join(config.checkpoint_dir, "best_model.pth"), logger)

            if epoch % config.save_every == 0:
                safe_save(checkpoint, os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"), logger)
        
        # Early stopping
        if epochs_without_improvement >= config.patience:
            logger.warning(f"Early stopping! No improvement for {config.patience} epochs.")
            break
        
        # Clear cache
        if 'cuda' in config.device:
            torch.cuda.empty_cache()
    
    # ==========================================================================
    # Training Complete
    # ==========================================================================
    total_time = time.time() - training_start
    
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Best validation Dice: {best_dice:.4f}")
    logger.info(f"Checkpoints: {config.checkpoint_dir}")
    logger.info(f"Logs: {log_file}")


if __name__ == "__main__":
    main()

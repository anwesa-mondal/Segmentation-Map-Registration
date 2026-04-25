"""
Run inference.py for OASIS_OAS1_{IDX}_MR1, IDX = 0001..0100.
Prints per-sample Mean Dice and overall average at the end.
"""

import subprocess
import json
from pathlib import Path

# =============================================================================
# CONFIGURE HERE
# =============================================================================

CHECKPOINT = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/training_mri_acm/20260220_161929/checkpoints/best_model.pth"
INPUT_MRI_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/scans/OASIS_OAS1_{idx}_MR1/brain.npy"
# INPUT_MRI_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/soham_data/fomo-60k/sub_{idx}/ses_1/t1.nii.gz"
INPUT_SEG_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/oasis_synthseg_output/output/OASIS_OAS1_{idx}_MR1/orig_synthseg.nii.gz"
# INPUT_SEG_TEMPLATE  = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/fomo60k_synthseg_data/output/output/sub_{idx}/ses_1/t1_synthseg.nii.gz"
OUTPUT_DIR_TEMPLATE = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/nishchay_results/OASIS_OAS1_{idx}_MR1"
# OUTPUT_DIR_TEMPLATE = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/nishchay_results_fomo60k/sub_{idx}/ses_1"
DEVICE = "cuda:4"

INFERENCE_SCRIPT = Path(__file__).parent / "inference.py"

RESUME_EPOCH = 38

# =============================================================================

dice_scores = {}
skipped = []

# for i in range(1, 41):
for i in range(RESUME_EPOCH, 100):
    idx = f"{i:04d}"
    # idx = i
    input_mri = INPUT_MRI_TEMPLATE.format(idx=idx)
    input_seg = INPUT_SEG_TEMPLATE.format(idx=idx)
    output_dir = OUTPUT_DIR_TEMPLATE.format(idx=idx)

    # Skip if input files don't exist
    if not Path(input_mri).exists() or not Path(input_seg).exists():
        print(f"[{idx}] SKIP — input files not found")
        skipped.append(idx)
        continue

    print(f"[{idx}] Running inference...", flush=True)
    result = subprocess.run(
        ["python", str(INFERENCE_SCRIPT),
         "--input", input_mri,
         "--input_seg", input_seg,
         "--checkpoint", CHECKPOINT,
         "--output_dir", output_dir,
         "--device", DEVICE,
         "--losses_only"],
        # capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[{idx}] FAILED\n{result.stderr[-500:]}")
        skipped.append(idx)
        continue

    losses_path = Path(output_dir) / "losses.json"
    if not losses_path.exists():
        print(f"[{idx}] FAILED — losses.json not found")
        skipped.append(idx)
        continue

    with open(losses_path) as f:
        losses = json.load(f)

    dice = losses.get("dice_score")
    if dice is None:
        print(f"[{idx}] FAILED — dice_score missing in losses.json")
        skipped.append(idx)
        continue

    dice_scores[idx] = dice
    print(f"[{idx}] Mean Dice: {dice:.4f}")

# Summary
print("\n" + "=" * 50)
print(f"Processed : {len(dice_scores)} / 100")
print(f"Skipped   : {len(skipped)} {skipped if skipped else ''}")
if dice_scores:
    avg = sum(dice_scores.values()) / len(dice_scores)
    best_idx  = max(dice_scores, key=dice_scores.get)
    worst_idx = min(dice_scores, key=dice_scores.get)
    print(f"Avg Dice  : {avg:.4f}")
    print(f"Best      : {best_idx}  ({dice_scores[best_idx]:.4f})")
    print(f"Worst     : {worst_idx}  ({dice_scores[worst_idx]:.4f})")
print("=" * 50)

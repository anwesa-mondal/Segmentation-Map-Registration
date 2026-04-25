"""
Side-by-side comparison of 3 brain segmentation .nii.gz files.
Auto-detects N classes. Shows axial / coronal / sagittal views.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import nibabel as nib

# =============================================================================
# CONFIGURE HERE
# =============================================================================

PATH1 = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/oasis_synthseg_output/output/OASIS_OAS1_0001_MR1/orig_synthseg.nii.gz"
PATH2 = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/synthseg_only5labels/OASIS_OAS1_0001_MR1/orig_synthseg.nii.gz_five_labels.nii.gz"
PATH3 = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/scans/OASIS_OAS1_0001_MR1/seg4.nii.gz"

LABELS = ["SynthSeg", "SynthSeg Only 5 Labels", "GT", "SynthSeg Mapped"]   # column titles
OUTPUT = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/synthseg_only5labels/OASIS_OAS1_0001_MR1/five_labels_comparison.png"

# Optional: {label_int: "name"}, e.g. {0: "Background", 1: "Cortex"}
CLASS_NAMES = {}

mapping = {
    0: 0, 24: 0,
    3: 1, 42: 1, 
    10: 2, 49: 2, 11: 3, 50: 2, 12: 2, 51: 2, 13: 2, 52: 2,
    17: 2, 53: 2, 18: 2, 54: 2, 26: 2, 58: 2, 60: 2, 8: 2, 47: 2,
    2: 3, 41: 3, 7: 3, 46: 3, 16: 3, 28: 3,
    4: 4, 43: 4, 5: 4, 44: 4, 14: 4, 15: 4,
}

# =============================================================================

_PALETTE = [
    "#111111", "#ff6b6b", "#ffa500", "#ffd700", "#4cc9f0",
    "#7b2d8b", "#2ecc71", "#e74c3c", "#3498db", "#9b59b6",
    "#1abc9c", "#e67e22", "#27ae60", "#2980b9", "#8e44ad",
    "#16a085", "#d35400", "#c0392b", "#7f8c8d", "#f39c12",
    "#0000ff", "#00ff00", "#ff00ff", "#00ffff", "#ff4500",
    "#da70d6", "#adff2f", "#ff1493", "#00bfff", "#ffdab9",
]


def load_nii(path):
    data = np.asarray(nib.load(str(path)).dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    return data.astype(np.int32)


def nonzero_center(vol, axis):
    nz = np.argwhere(vol > 0)
    return int(np.median(nz[:, axis])) if len(nz) else vol.shape[axis] // 2


def seg_to_rgba(sl, label_to_idx, palette):
    rgba = np.zeros((*sl.shape, 4), dtype=np.float32)
    for lv, idx in label_to_idx.items():
        mask = sl == lv
        r, g, b, _ = mcolors.to_rgba(palette[idx % len(palette)])
        rgba[mask, :3] = (r, g, b)
        rgba[mask, 3] = 0.25 if lv == 0 else 0.85
    return rgba


# Load
vol1, vol2, vol3 = [load_nii(p) for p in [PATH1, PATH2, PATH3]]

# Apply mapping to vol1 to produce the 4th column
vol1_mapped = np.zeros_like(vol1)
for src, dst in mapping.items():
    vol1_mapped[vol1 == src] = dst
# Any label not in mapping stays 0 (background)

vols = [vol1, vol2, vol3, vol1_mapped]

# Auto-detect classes across all 4 volumes
all_classes = sorted(set().union(*[np.unique(v).tolist() for v in vols]))
n = len(all_classes)
palette = _PALETTE[:n] if n <= len(_PALETTE) else _PALETTE + [
    mcolors.to_hex(mcolors.hsv_to_rgb([(i / (n - len(_PALETTE))) * 0.8, 0.8, 0.9]))
    for i in range(n - len(_PALETTE))
]
label_to_idx = {lv: i for i, lv in enumerate(all_classes)}

# Smart slices (tissue centre-of-mass)
d = nonzero_center(vol1, 0)
h = nonzero_center(vol1, 1)
w = nonzero_center(vol1, 2)

# Plot
plt.style.use("dark_background")
fig = plt.figure(figsize=(28, 21))
gs = GridSpec(3, 4, figure=fig, hspace=0.06, wspace=0.04)

view_names = ["Axial", "Coronal", "Sagittal"]
get_slice = [
    lambda v: v[d],
    lambda v: v[:, h, :],
    lambda v: v[:, :, w],
]

for row in range(3):
    for col, (vol, col_label) in enumerate(zip(vols, LABELS)):
        sl = get_slice[row](vol)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(seg_to_rgba(sl, label_to_idx, palette),
                  aspect="auto", interpolation="nearest")
        if row == 0:
            ax.set_title(col_label, fontsize=15, fontweight="bold", pad=8)
        if col == 0:
            ax.set_ylabel(view_names[row], fontsize=13, labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])

# Legend
patches = [
    mpatches.Patch(color=palette[i],
                   label=f"{lv}: {CLASS_NAMES.get(lv, 'Background' if lv == 0 else f'Class {lv}')}")
    for i, lv in enumerate(all_classes)
]
fig.legend(handles=patches, loc="lower center", ncol=min(n, 8),
           fontsize=11, framealpha=0.85, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(f"Brain Segmentation Comparison  |  {n} classes",
             fontsize=17, fontweight="bold", y=0.995)

fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print(f"Saved → {OUTPUT}")
plt.show()

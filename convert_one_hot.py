import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm


labels = [0, 1, 2, 3, 4]

def to_one_hot(segmentation, labels):
    one_hot = np.zeros((len(labels), *segmentation.shape), dtype=np.uint8)
    for i, label in enumerate(labels):
        one_hot[i] = (segmentation == label).astype(np.uint8)
    return one_hot

parser = argparse.ArgumentParser(description="Convert segmentation maps to 5-channel one-hot .npy")
parser.add_argument(
    "--input_txt",
    type=str,
    required=True,
    help="Text file with paths to segmentation .nii.gz files"
)
parser.add_argument(
    "--output_name",
    type=str,
    default="seg5_onehot.npy",
    help="Output filename to save in each subject directory"
)

args = parser.parse_args()


with open(args.input_txt, "r") as f:
    paths = [p.strip() for p in f.readlines() if p.strip()]


for path in tqdm(paths, desc="Processing segmentation maps"):
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        continue

    img = nib.load(path)
    seg_map = img.get_fdata().astype(np.int16)

    one_hot_map = to_one_hot(seg_map, labels)

    subject_dir = os.path.dirname(path)
    output_path = os.path.join(subject_dir, args.output_name)

    np.save(output_path, one_hot_map)

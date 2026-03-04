import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from model import UNetCore, IterativeRegNet, SpatialTransformer
from get_data import SegDataset
from losses import dice_loss


def jacobian_det(flow):
    dx = flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1]
    dy = flow[:, :, :-1, 1:, :-1] - flow[:, :, :-1, :-1, :-1]
    dz = flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1]

    J = torch.zeros(flow.size(0), dx.size(2), dx.size(3), dx.size(4), 3, 3, device=flow.device)
    J[..., 0, 0] = dx[:, 0]
    J[..., 0, 1] = dy[:, 0]
    J[..., 0, 2] = dz[:, 0]
    J[..., 1, 0] = dx[:, 1]
    J[..., 1, 1] = dy[:, 1]
    J[..., 1, 2] = dz[:, 1]
    J[..., 2, 0] = dx[:, 2]
    J[..., 2, 1] = dy[:, 2]
    J[..., 2, 2] = dz[:, 2]

    return (
        J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1])
        - J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 0])
        + J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1] - J[..., 1, 1] * J[..., 2, 0])
    )


def evaluate(model, stn, loader, mode):
    model.eval()
    dice_vals = []
    neg_jac_vals = []

    with torch.no_grad():
        for moving, fixed in tqdm(loader, leave=False):
            moving = moving.to(DEVICE)
            fixed = fixed.to(DEVICE)

            if mode == "single":
                x = torch.cat([moving, fixed], dim=1)
                flow = model(x)
            else:
                flow = model(moving, fixed, stn)

            warped = stn(moving, flow)
            dice_vals.append(1.0 - dice_loss(warped, fixed).item())


    plt.ylabel("Negative Jacobian Ratio")
                parser = argparse.ArgumentParser(description="Test multiple registration models with CLI-specified configs.")
                parser.add_argument('--test_txt', type=str, required=True, help='Path to test list file')
                parser.add_argument('--template', type=str, required=True, help='Path to template .npy file')
                parser.add_argument('--output_dir', type=str, default='test_viz', help='Directory to save results')
                parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
                parser.add_argument('--wandb_project', type=str, default='seg-deformation', help='wandb project name')
                parser.add_argument('--wandb_run', type=str, default='test_evaluation', help='wandb run name')
                parser.add_argument('--model_cfgs', type=str, nargs='+', required=True,
                                    help='List of model configs as: name:path:mode:steps (e.g. singleK2:weights/final.pth:single:1)')
                args = parser.parse_args()

                wandb.init(project=args.wandb_project, name=args.wandb_run)
                os.makedirs(args.output_dir, exist_ok=True)

                dataset = SegDataset(args.test_txt, args.template)
                loader = DataLoader(dataset, batch_size=1, shuffle=False)
                stn = SpatialTransformer((128, 128, 128), device=args.device).to(args.device)

                results = []

                for cfg_str in args.model_cfgs:
                    # Parse config string
                    # Format: name:path:mode:steps
                    parts = cfg_str.split(':')
                    if len(parts) != 4:
                        raise ValueError(f"Model config '{cfg_str}' must be in format name:path:mode:steps")
                    name, path, mode, steps = parts
                    steps = int(steps)

                    if mode == "single":
                        model = UNetCore(10, 3).to(args.device)
                    else:
                        model = IterativeRegNet(10, 3, steps=steps, shared=(mode == "iter_shared")).to(args.device)

                    model.load_state_dict(torch.load(path, map_location=args.device))

                    mean_dice, neg_jac = evaluate(model, stn, loader, mode)

                    wandb.log({
                        f"{name}/mean_dice": mean_dice,
                        f"{name}/neg_jac_ratio": neg_jac
                    })

                    results.append({
                        "model": name,
                        "mean_dice": mean_dice,
                        "neg_jacobian_ratio": neg_jac
                    })

                df = pd.DataFrame(results)
                df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)

                plt.figure(figsize=(8, 4))
                plt.bar(df["model"], df["mean_dice"])
                plt.ylabel("Mean Dice")
                plt.xticks(rotation=15)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "dice_comparison.png"))

                plt.figure(figsize=(8, 4))
                plt.bar(df["model"], df["neg_jacobian_ratio"])
                plt.ylabel("Negative Jacobian Ratio")
                plt.xticks(rotation=15)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "jacobian_comparison.png"))

                wandb.finish()
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("test_viz/jacobian_comparison.png")

    wandb.finish()


if __name__ == "__main__":
    main()

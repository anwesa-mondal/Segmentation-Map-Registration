import torch
from torch.utils.data import DataLoader
from model import UNetCore, IterativeRegNet, SpatialTransformer
from get_data import SegDataset
from losses import composite_loss
from tqdm import tqdm
import argparse
import os
import wandb
from torch.cuda.amp import GradScaler
from datetime import datetime

wandb.init(project="seg-deformation")
torch.cuda.empty_cache()


def train_epoch(model, stn, loader, optimizer, scaler, scheduler, device, mode):
    model.train()
    total = 0.0

    for moving, fixed in tqdm(loader, leave=False):
        moving = moving.to(device)
        fixed = fixed.to(device)

        optimizer.zero_grad()

        if mode == "single":
            x = torch.cat([moving, fixed], dim=1)
            flow = model(x)
        else:
            flow = model(moving, fixed, stn)

        warped = stn(moving, flow)
        loss = composite_loss(warped, fixed, flow)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item()

    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--template_path", required=True)
    parser.add_argument("--mode", choices=["single", "iter_shared", "iter_unshared"], default="single")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_root", default="./weights")
    args = parser.parse_args()

    device = "cuda:0"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_root}_{args.mode}_K{args.steps}_{ts}"
    os.makedirs(save_dir, exist_ok=True)

    dataset = SegDataset(args.train_txt, args.template_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    stn = SpatialTransformer((128, 128, 128), device=device).to(device)

    if args.mode == "single":
        model = UNetCore(10, 3).to(device)
    elif args.mode == "iter_shared":
        model = IterativeRegNet(10, 3, steps=args.steps, shared=True).to(device)
    else:
        model = IterativeRegNet(10, 3, steps=args.steps, shared=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )
    scaler = GradScaler()

    for epoch in range(args.epochs):
        loss = train_epoch(
            model, stn, loader,
            optimizer, scaler, scheduler,
            device, args.mode
        )

        wandb.log({"epoch": epoch + 1, "loss": loss, "mode": args.mode})
        print(f"[{args.mode}] Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")

    torch.save(model.state_dict(), f"{save_dir}/final.pth")
    print(f"[DONE] Saved to {save_dir}")


if __name__ == "__main__":
    main()

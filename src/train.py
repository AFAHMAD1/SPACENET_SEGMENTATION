import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config as C
from dataset import SpaceNetDataset

NUM_CLASSES = 3
MODELS = {
    "unet": lambda: smp.Unet("resnet34", encoder_weights="imagenet", classes=NUM_CLASSES),
    "deeplabv3plus": lambda: smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", classes=NUM_CLASSES),
    "segformer": lambda: smp.Segformer("mit_b1", encoder_weights="imagenet", classes=NUM_CLASSES),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice, both ignoring 255."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=C.IGNORE)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=C.IGNORE)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)


def confusion_matrix(pred, target):
    valid = target != C.IGNORE
    idx = target[valid] * NUM_CLASSES + pred[valid]
    return torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).cpu().numpy()


def metrics_from_cm(cm):
    cm = cm.astype("float64")
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return {"iou": iou, "miou": float(np.nanmean(iou)), "acc": float(tp.sum() / max(cm.sum(), 1))}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    cm, total, n = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype="int64"), 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits, y).item() * x.size(0)
        n += x.size(0)
        cm += confusion_matrix(logits.argmax(1), y)
    return total / n, metrics_from_cm(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data_dir", default=str(C.ROOT / "data"), help="folder with images/, masks/, splits.csv")
    ap.add_argument("--out_dir", default=str(C.ROOT), help="models/ and results/ are created here")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="use only N tiles per split (for testing)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    ckpt, log_path = out / "models" / f"{a.model}_best.pth", out / "results" / f"{a.model}_log.csv"

    train_ds, val_ds = SpaceNetDataset("train", data_dir=a.data_dir), SpaceNetDataset("val", data_dir=a.data_dir)
    if a.limit:
        train_ds, val_ds = Subset(train_ds, range(min(a.limit, len(train_ds)))), Subset(val_ds, range(min(a.limit, len(val_ds))))
    g = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, drop_last=True, num_workers=a.num_workers,
                          generator=g, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, a.batch_size, shuffle=False, num_workers=a.num_workers)

    model = MODELS[a.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.model}: {n_params / 1e6:.2f}M parameters | device {device} | train {len(train_ds)} val {len(val_ds)}")

    loss_fn = CombinedLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cols = ["epoch", "train_loss", "val_loss", "iou_land", "iou_road", "iou_building", "miou", "pixel_acc", "lr", "epoch_time_s"]
    best, bad_epochs = -1.0, 0
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for epoch in range(1, a.epochs + 1):
            t0 = time.time()
            model.train()
            total, n = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = total / n
            lr = opt.param_groups[0]["lr"]
            sched.step()
            val_loss, m = evaluate(model, val_dl, loss_fn, device)
            dt = time.time() - t0
            w.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", *[f"{v:.4f}" for v in m["iou"]],
                        f"{m['miou']:.4f}", f"{m['acc']:.4f}", f"{lr:.6f}", f"{dt:.1f}"])
            f.flush()
            print(f"ep {epoch:02d} train {train_loss:.4f} val {val_loss:.4f} IoU land/road/bldg "
                  f"{m['iou'][0]:.3f}/{m['iou'][1]:.3f}/{m['iou'][2]:.3f} mIoU {m['miou']:.4f} acc {m['acc']:.4f} {dt:.0f}s")
            if not np.isfinite(train_loss):
                raise RuntimeError("train loss is not finite")
            if m["miou"] > best:
                best, bad_epochs = m["miou"], 0
                torch.save({"model": a.model, "state_dict": model.state_dict(), "epoch": epoch, "val_miou": best}, ckpt)
            else:
                bad_epochs += 1
                if bad_epochs >= a.patience:
                    print(f"early stopping at epoch {epoch} (best mIoU {best:.4f})")
                    break
    print(f"best val mIoU {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader

import config as C
from check_masks import overlay
from dataset import SpaceNetDataset

RES = C.ROOT / "results"
MODELS_DIR = C.ROOT / "models"
NAMES = {"unet": "U-Net", "deeplabv3plus": "DeepLabV3+", "segformer": "SegFormer"}
CLASSES = list(C.CLASS_NAMES.values())
BUILD = {  # same architectures as train.py, but no pretrained download: weights come from the checkpoint
    "unet": lambda: smp.Unet("resnet34", encoder_weights=None, classes=3),
    "deeplabv3plus": lambda: smp.DeepLabV3Plus("resnet34", encoder_weights=None, classes=3),
    "segformer": lambda: smp.Segformer("mit_b1", encoder_weights=None, classes=3),
}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def find_ckpt(name):
    for ext in (".pth", ".zip"):  # Kaggle download renamed them to .zip; torch.load doesn't care
        p = MODELS_DIR / f"{name}_best{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def load_model(name):
    ck = torch.load(find_ckpt(name), map_location="cpu", weights_only=True)
    m = BUILD[name]()
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def cm_of(pred, gt):
    v = gt != C.IGNORE
    return np.bincount(gt[v].astype(int) * 3 + pred[v], minlength=9).reshape(3, 3)


def summarize(cm):
    cm = cm.astype("float64")
    tp, fp, fn = np.diag(cm), cm.sum(0) - np.diag(cm), cm.sum(1) - np.diag(cm)
    with np.errstate(all="ignore"):
        iou, dice = tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)
        prec, rec = tp / (tp + fp), tp / (tp + fn)
    return iou, dice, prec, rec, tp.sum() / cm.sum()


def training_log_summary(name):
    log = pd.read_csv(RES / f"{name}_log.csv")
    b = log.loc[log.miou.idxmax()]
    return {"best_val_mIoU": b.miou, "best_epoch": int(b.epoch), "epochs_run": len(log),
            "train_time_min": log.epoch_time_s.sum() / 60}, log


def plot_curves(logs):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for name, log in logs.items():
        ax[0].plot(log.epoch, log.train_loss, label=NAMES[name])
        ax[1].plot(log.epoch, log.val_loss, label=NAMES[name])
        ax[2].plot(log.epoch, log.miou, label=NAMES[name])
    for a, t in zip(ax, ["Train loss", "Val loss", "Val mIoU"]):
        a.set_title(t)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(RES / "training_curves.png", dpi=110)
    plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    (RES / "predictions").mkdir(exist_ok=True)
    torch.manual_seed(C.SEED)

    # ---- 1. training logs
    train_info, logs = {}, {}
    for n in NAMES:
        train_info[n], logs[n] = training_log_summary(n)
    plot_curves(logs)

    # ---- 2. test-set predictions (center crop 512, as in dataset.py)
    ds = SpaceNetDataset("test")
    ids = ds.ids
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    gts = np.concatenate([y.numpy().astype("uint8") for _, y in dl])
    preds, rows, cms = {}, [], {}
    for n in NAMES:
        model, ck = load_model(n)
        out = []
        with torch.no_grad():
            for x, _ in dl:
                out.append(model(x).argmax(1).numpy().astype("uint8"))
        preds[n] = np.concatenate(out)
        cm = sum(cm_of(p, g) for p, g in zip(preds[n], gts))
        cms[n] = cm
        iou, dice, prec, rec, acc = summarize(cm)
        r = {"model": NAMES[n]}
        for i, c in enumerate(CLASSES):
            r.update({f"IoU_{c}": iou[i], f"Dice_{c}": dice[i], f"Precision_{c}": prec[i], f"Recall_{c}": rec[i]})
        r.update({"test_mIoU": iou.mean(), "test_mean_Dice": dice.mean(), "pixel_acc": acc})

        # ---- 3. efficiency
        r["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        r["ckpt_MB"] = find_ckpt(n).stat().st_size / 1e6
        x1 = ds[0][0][None]
        with torch.no_grad():
            for _ in range(2):
                model(x1)  # warm-up
            t = []
            for i in range(15):
                x1 = ds[i][0][None]
                t0 = time.perf_counter()
                model(x1)
                t.append(time.perf_counter() - t0)
        r["cpu_ms_per_image"] = 1000 * float(np.mean(t))
        r.update(train_info[n])
        rows.append(r)

        # confusion matrix figure (rows = ground truth, normalized)
        norm = cm / cm.sum(1, keepdims=True)
        fig, a = plt.subplots(figsize=(4.5, 4))
        a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
        a.set_xticks(range(3), CLASSES)
        a.set_yticks(range(3), CLASSES)
        a.set_xlabel("predicted")
        a.set_ylabel("ground truth")
        a.set_title(f"{NAMES[n]} (test, row-normalized)")
        fig.tight_layout()
        fig.savefig(RES / f"confusion_{n}.png", dpi=110)
        plt.close(fig)
        print(n, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "metrics.csv", index=False, float_format="%.4f")
    write_md(df)

    # ---- 5. prediction figures
    per_tile = pd.DataFrame({"tile_id": ids})
    valid = gts != C.IGNORE
    for n in NAMES:
        m = []
        for p, g in zip(preds[n], gts):
            iou = summarize(cm_of(p, g))[0]
            m.append(np.nanmean(iou) if np.isfinite(iou).any() else np.nan)
        per_tile[f"mIoU_{n}"] = m
    pairs = [("unet", "deeplabv3plus"), ("unet", "segformer"), ("deeplabv3plus", "segformer")]
    dis = np.zeros(len(ids))
    for i in range(len(ids)):
        v = valid[i]
        dis[i] = np.mean([(preds[a][i][v] != preds[b][i][v]).mean() for a, b in pairs]) if v.any() else 0
    per_tile["disagreement"] = dis
    lab = np.array([((g == 1) | (g == 2)).sum() / max((g != C.IGNORE).sum(), 1) for g in gts])
    per_tile["labelled_frac"] = lab
    best = df.loc[df.test_mIoU.idxmax(), "model"]
    best_key = [k for k, v in NAMES.items() if v == best][0]
    per_tile.to_csv(RES / "per_tile_test.csv", index=False, float_format="%.4f")

    chosen, reasons = [], {}

    def take(order, k, why):
        got = 0
        for i in order:
            if i not in reasons and got < k:
                reasons[i] = why
                chosen.append(i)
                got += 1

    take(np.argsort(-dis), 3, "models disagree most")
    take(np.argsort(per_tile[f"mIoU_{best_key}"].fillna(9).values), 3, f"{best} does worst")
    take(np.argsort(-lab), 2, "dense")
    take(np.argsort(lab), 2, "sparse")

    for i in chosen:
        img = denorm_img(ds, i)
        g = gts[i]
        fig, ax = plt.subplots(1, 5, figsize=(25, 5.4))
        ax[0].imshow(img)
        ax[0].set_title(f"{ids[i]}\n({reasons[i]})")
        ax[1].imshow(overlay(img, g))
        ax[1].set_title("ground truth")
        for k, n in enumerate(NAMES):
            ax[2 + k].imshow(overlay(img, preds[n][i]))
            ax[2 + k].set_title(f"{NAMES[n]}  mIoU {per_tile.loc[i, f'mIoU_{n}']:.2f}")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(RES / "predictions" / f"{ids[i]}.png", dpi=60)
        plt.close(fig)
    print(pd.DataFrame({"tile": [ids[i] for i in chosen], "why": [reasons[i] for i in chosen]}).to_string())


def denorm_img(ds, i):
    x = ds[i][0].numpy().transpose(1, 2, 0)
    return (np.clip(x * STD + MEAN, 0, 1) * 255).astype("uint8")


def write_md(df):
    def f(v, nd=3):
        return f"{v:.{nd}f}"

    L = ["# Test-set comparison (92 tiles, 512x512 center crop, 255 ignored)\n",
         "| Model | mIoU | mean Dice | Pixel acc | IoU land | IoU road | IoU building |", "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.model} | {f(r.test_mIoU)} | {f(r.test_mean_Dice)} | {f(r.pixel_acc)} | {f(r.IoU_land)} | {f(r.IoU_road)} | {f(r.IoU_building)} |")
    L += ["\n## Per-class Dice / precision / recall\n", "| Model | Class | Dice | Precision | Recall |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        for c in CLASSES:
            L.append(f"| {r.model} | {c} | {f(r[f'Dice_{c}'])} | {f(r[f'Precision_{c}'])} | {f(r[f'Recall_{c}'])} |")
    L += ["\n## Training and efficiency\n",
          "| Model | Best val mIoU | Best epoch | Epochs run | Train time (min, Kaggle T4) | Params (M) | Checkpoint (MB) | CPU ms/image* |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.model} | {f(r.best_val_mIoU, 4)} | {int(r.best_epoch)} | {int(r.epochs_run)} | {f(r.train_time_min, 1)} | {f(r.params_M, 2)} | {f(r.ckpt_MB, 1)} | {f(r.cpu_ms_per_image, 0)} |")
    L.append("\n*Inference time is on this machine's CPU (batch of 1, 512x512, mean of 15 images, after warm-up), not GPU time.")
    (RES / "metrics.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

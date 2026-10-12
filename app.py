import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config as C
from check_masks import overlay
from evaluate import MEAN, NAMES, STD, cm_of, load_model, summarize

SIZE = 512
torch.set_num_threads(max(1, torch.get_num_threads()))
MODELS = {k: load_model(k)[0] for k in NAMES}  # loaded once, CPU
TEST_IDS = list(pd.read_csv(C.SPLITS_CSV).query("split == 'test'").tile_id)

LEGEND = """<div style="display:flex;gap:24px;align-items:center;font-size:15px">
<b>Legend:</b>
<span><span style="display:inline-block;width:14px;height:14px;background:#ffff00;border:1px solid #888"></span> road</span>
<span><span style="display:inline-block;width:14px;height:14px;background:#ff0000;border:1px solid #888"></span> building</span>
<span><span style="display:inline-block;width:14px;height:14px;background:transparent;border:1px solid #888"></span> land (no color)</span>
<span><span style="display:inline-block;width:14px;height:14px;background:#808080;border:1px solid #888"></span> no-data (ignored, ground truth only)</span>
</div>"""


def center_crop(img):
    """Center-crop 512x512 (test tiles are 650x650), as in evaluation."""
    w, h = img.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return img.crop((l, t, l + SIZE, t + SIZE))


STRIDE = 384  # 128 px overlap between neighbouring windows
BATCH = 4
_ramp = np.minimum(1.0, np.minimum(np.arange(SIZE) + 1, SIZE - np.arange(SIZE)) / 64.0)
WEIGHT = torch.from_numpy(np.outer(_ramp, _ramp)).float()  # down-weights window borders when blending


def pad_to_min(img):
    """Pad each side up to 512 if smaller; returns the padded image (original sits at top-left)."""
    h, w = img.shape[:2]
    ph, pw = max(0, SIZE - h), max(0, SIZE - w)
    if ph == 0 and pw == 0:
        return img
    mode = "symmetric" if ph <= h and pw <= w else "edge"
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode=mode)


def starts(n):
    s = list(range(0, n - SIZE + 1, STRIDE))
    if s[-1] != n - SIZE:
        s.append(n - SIZE)
    return s


@torch.no_grad()
def segment(model, img):
    """Sliding-window prediction at full size. img uint8 (H,W,3) with H,W >= 512 -> uint8 (H,W) class map."""
    H, W = img.shape[:2]
    x = torch.from_numpy(((img / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()  # same as dataset.py
    probs, norm = torch.zeros(3, H, W), torch.zeros(H, W)
    wins = [(t, l) for t in starts(H) for l in starts(W)]
    for i in range(0, len(wins), BATCH):
        chunk = wins[i:i + BATCH]
        p = model(torch.stack([x[:, t:t + SIZE, l:l + SIZE] for t, l in chunk])).softmax(1)
        for (t, l), pi in zip(chunk, p):
            probs[:, t:t + SIZE, l:l + SIZE] += pi * WEIGHT
            norm[t:t + SIZE, l:l + SIZE] += WEIGHT
    return (probs / norm).argmax(0).numpy().astype("uint8"), len(wins)


def predict(img, gt=None):
    """img: uint8 (H,W,3), any size. Returns 3 overlays (original size) and 3 stats texts."""
    H, W = img.shape[:2]
    padded = pad_to_min(img)
    overlays, stats = [], []
    for k, model in MODELS.items():
        t0 = time.perf_counter()
        pred, n_win = segment(model, padded)
        ms = 1000 * (time.perf_counter() - t0)
        pred = pred[:H, :W]  # drop the padding
        frac = np.bincount(pred.ravel(), minlength=3) / pred.size * 100
        s = (f"**{NAMES[k]}**  \nroad {frac[1]:.1f}% · building {frac[2]:.1f}% · land {frac[0]:.1f}%  "
             f"\ninference {ms:.0f} ms (CPU, {n_win} window{'s' if n_win > 1 else ''})")
        if gt is not None:
            iou = summarize(cm_of(pred, gt))[0]
            s += f"  \nmIoU on this tile: {np.nanmean(iou):.3f}"
        overlays.append(overlay(img, pred))
        stats.append(s)
    return overlays, stats


def outputs(img, gt, overlays, stats):
    return [img,
            gr.Image(value=overlay(img, gt) if gt is not None else None, visible=gt is not None),
            *overlays, *stats]


def run_tile(tile_id):
    if not tile_id:
        return gr.skip()
    full = Image.open(C.IMAGES_DIR / f"{tile_id}.png").convert("RGB")
    mask = np.array(Image.open(C.MASKS_DIR / f"{tile_id}.png"))
    h, w = mask.shape
    t, l = (h - SIZE) // 2, (w - SIZE) // 2  # same center crop as dataset.py; mask is cropped, never resampled
    img, gt = np.array(center_crop(full)), mask[t:t + SIZE, l:l + SIZE]
    overlays, stats = predict(img, gt)
    return outputs(img, gt, overlays, stats)


def run_upload(pil):
    if pil is None:
        return gr.skip()
    img = np.array(pil.convert("RGB"))  # no resize, no crop: predicted at full size
    overlays, stats = predict(img)
    return outputs(img, None, overlays, stats)


with gr.Blocks(title="SpaceNet Paris segmentation") as demo:
    gr.Markdown("# Land / road / building segmentation\nU-Net, DeepLabV3+ and SegFormer trained on SpaceNet Paris. "
                "Pick a **test tile** (ground truth is shown) or **upload** any RGB aerial/satellite image "
                "(any size: large images are processed in overlapping 512×512 windows and stitched, small ones are padded). "
                "Images from other cities or zoom levels will generally work worse.")
    gr.HTML(LEGEND)
    with gr.Row():
        tile = gr.Dropdown(TEST_IDS, label="Test tile", value=None)
        upload = gr.Image(type="pil", label="…or upload an image", height=200)
    with gr.Row():
        orig = gr.Image(label="Original", interactive=False)
        gt_img = gr.Image(label="Ground truth", interactive=False, visible=False)
    with gr.Row():
        pred_imgs, pred_txt = [], []
        for k in NAMES:
            with gr.Column():
                pred_imgs.append(gr.Image(label=NAMES[k], interactive=False))
                pred_txt.append(gr.Markdown())
    outs = [orig, gt_img, *pred_imgs, *pred_txt]
    tile.change(run_tile, tile, outs, api_name="tile")
    upload.change(run_upload, upload, outs, api_name="upload")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

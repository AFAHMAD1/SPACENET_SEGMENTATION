# Test-set comparison (92 tiles, 512x512 center crop, 255 ignored)

| Model | mIoU | mean Dice | Pixel acc | IoU land | IoU road | IoU building |
|---|---|---|---|---|---|---|
| U-Net | 0.725 | 0.831 | 0.947 | 0.942 | 0.565 | 0.670 |
| DeepLabV3+ | 0.724 | 0.831 | 0.947 | 0.942 | 0.580 | 0.651 |
| SegFormer | 0.725 | 0.831 | 0.945 | 0.940 | 0.577 | 0.657 |

## Per-class Dice / precision / recall

| Model | Class | Dice | Precision | Recall |
|---|---|---|---|---|
| U-Net | land | 0.970 | 0.969 | 0.971 |
| U-Net | road | 0.722 | 0.737 | 0.707 |
| U-Net | building | 0.802 | 0.796 | 0.809 |
| DeepLabV3+ | land | 0.970 | 0.969 | 0.971 |
| DeepLabV3+ | road | 0.735 | 0.752 | 0.718 |
| DeepLabV3+ | building | 0.789 | 0.790 | 0.787 |
| SegFormer | land | 0.969 | 0.971 | 0.967 |
| SegFormer | road | 0.732 | 0.720 | 0.744 |
| SegFormer | building | 0.793 | 0.784 | 0.803 |

## Training and efficiency

| Model | Best val mIoU | Best epoch | Epochs run | Train time (min, Kaggle T4) | Params (M) | Checkpoint (MB) | CPU ms/image* |
|---|---|---|---|---|---|---|---|
| U-Net | 0.7271 | 54 | 60 | 15.2 | 24.44 | 97.9 | 386 |
| DeepLabV3+ | 0.7203 | 51 | 60 | 12.3 | 22.44 | 89.9 | 525 |
| SegFormer | 0.7198 | 33 | 43 | 10.4 | 13.68 | 54.8 | 428 |

*Inference time is on this machine's CPU (batch of 1, 512x512, mean of 15 images, after warm-up), not GPU time.

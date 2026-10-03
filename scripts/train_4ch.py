import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


def upgrade_first_conv(model, in_channels: int = 4):
    target_name = None
    target_module = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            target_name = name
            target_module = module
            break

    if target_module is None:
        raise RuntimeError("Could not find a Conv2d layer with 3 input channels.")

    new_conv = torch.nn.Conv2d(
        in_channels,
        target_module.out_channels,
        kernel_size=target_module.kernel_size,
        stride=target_module.stride,
        padding=target_module.padding,
        dilation=target_module.dilation,
        groups=target_module.groups,
        bias=target_module.bias is not None,
        padding_mode=target_module.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = target_module.weight
        gray = (
            target_module.weight[:, 0, :, :] * 0.2989
            + target_module.weight[:, 1, :, :] * 0.5870
            + target_module.weight[:, 2, :, :] * 0.1140
        )
        new_conv.weight[:, 3, :, :] = gray
        if target_module.bias is not None:
            new_conv.bias.copy_(target_module.bias)

    # Replace module
    parent = model
    name_parts = target_name.split(".")
    for part in name_parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, name_parts[-1], new_conv)


def patch_load_image(blackhat_root: Path, kernel_size: int):
    import ultralytics.data.base as udb

    if getattr(udb.BaseDataset.load_image, "_is_4ch", False):
        return

    original_load = udb.BaseDataset.load_image

    def load_image_4ch(self, i, rect_mode=True):
        im, (h0, w0), (h, w) = original_load(self, i, rect_mode=rect_mode)
        im_file = Path(self.im_files[i])
        split = im_file.parent.name
        bh_path = blackhat_root / f"k{kernel_size}" / split / f"{im_file.stem}.png"
        if bh_path.exists():
            bh = cv2.imread(str(bh_path), cv2.IMREAD_GRAYSCALE)
        else:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        if bh.shape[:2] != im.shape[:2]:
            bh = cv2.resize(bh, (im.shape[1], im.shape[0]))

        im4 = np.dstack([im, bh])
        return im4, (h0, w0), (h, w)

    load_image_4ch._is_4ch = True
    udb.BaseDataset.load_image = load_image_4ch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("configs/rdd2022_4ch.yaml"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/rdd2022_full")
    parser.add_argument("--name", type=str, default="blackhat_4ch_k17_blur_full_e40")
    parser.add_argument("--blackhat-root", type=Path, default=Path("data/rdd2022_yolo/blackhat_blur"))
    parser.add_argument("--kernel-size", type=int, default=17)
    parser.add_argument("--weights", type=Path, default=Path("yolov8s.pt"))
    args = parser.parse_args()

    patch_load_image(args.blackhat_root, args.kernel_size)

    # Start from the pretrained RGB model, then expand its input layer.  The
    # RGB filters are retained and the morphology filter is initialized from
    # their luminance-weighted average.
    model = YOLO(str(args.weights))
    upgrade_first_conv(model.model)
    # Ensure trainer uses the prepared 4-channel model.
    model.model.yaml["ch"] = 4
    model.model.args["ch"] = 4

    first_conv = model.model.model[0].conv
    print(f"first_conv.in_channels={first_conv.in_channels}")

    class FourChannelTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return model.model

    # workers=0 is CRITICAL: keeps train+val dataloaders in the main process
    # so the load_image monkey-patch stays active during validation.
    # Multiprocessing workers spawn fresh processes that don't inherit the patch,
    # causing "expected 4 channels, got 3" during val.
    model.train(
        trainer=FourChannelTrainer,
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=0,       # <-- KEY FIX: patch survives in main process only
        exist_ok=True,   # allow rerun without renaming
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        flipud=0.0,
    )


if __name__ == "__main__":
    main()

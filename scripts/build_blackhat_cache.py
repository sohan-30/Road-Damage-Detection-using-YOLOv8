import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compute_blackhat(gray: np.ndarray, kernel_size: int, use_gaussian: bool) -> np.ndarray:
    if use_gaussian:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return blackhat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, default=Path("data/rdd2022_yolo/images"))
    parser.add_argument("--output-root", type=Path, default=Path("data/rdd2022_yolo/blackhat"))
    parser.add_argument("--kernel-size", type=int, default=17)
    parser.add_argument("--use-gaussian", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "debug"])
    args = parser.parse_args()

    output_dir = args.output_root / f"k{args.kernel_size}"

    for split in args.splits:
        split_dir = args.images_root / split
        if not split_dir.exists():
            continue
        out_split = output_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(split_dir.glob("*.jpg"))
        for image_path in tqdm(image_paths, desc=f"{split} k={args.kernel_size}"):
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blackhat = compute_blackhat(gray, args.kernel_size, args.use_gaussian)
            out_path = out_split / (image_path.stem + ".png")
            cv2.imwrite(str(out_path), blackhat)


if __name__ == "__main__":
    main()

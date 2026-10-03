"""
train_full_comparison.py
========================
Trains 3-ch baseline and 4-ch BH-k17+Gaussian-blur models on the FULL dataset
(China_Drone + China_MotorBike, 3502 train / 437 val), then runs convergence
analysis to find the optimal epoch at which 4-ch outperforms 3-ch.

Usage:
    python scripts/train_full_comparison.py [--epochs 100] [--batch 8] [--device 0]

The 4-ch model trains AFTER the 3-ch model finishes (sequential to avoid OOM).
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=str(BASE))
    if result.returncode != 0:
        print(f"\n[ERROR] '{desc}' failed with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--device",  type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/rdd2022_full")
    parser.add_argument(
        "--skip-3ch", action="store_true",
        help="Skip 3-ch training (if already done)"
    )
    parser.add_argument(
        "--skip-4ch", action="store_true",
        help="Skip 4-ch training (if already done)"
    )
    args = parser.parse_args()

    py = sys.executable

    # ── 1. Train 3-ch baseline ─────────────────────────────────────────────
    if not args.skip_3ch:
        run(
            [
                py, "scripts/train_baseline.py",
                "--data",    "configs/rdd2022_3ch.yaml",
                "--epochs",  str(args.epochs),
                "--batch",   str(args.batch),
                "--imgsz",   str(args.imgsz),
                "--device",  args.device,
                "--project", args.project,
                "--name",    f"baseline_3ch_full_e{args.epochs}",
            ],
            f"Training 3-ch baseline — {args.epochs} epochs, batch={args.batch}, imgsz={args.imgsz}",
        )
    else:
        print("[SKIP] 3-ch training skipped.")

    # ── 2. Train 4-ch BH k17+Gaussian blur ────────────────────────────────
    if not args.skip_4ch:
        run(
            [
                py, "scripts/train_4ch.py",
                "--data",         "configs/rdd2022_4ch.yaml",
                "--epochs",       str(args.epochs),
                "--batch",        str(args.batch),
                "--imgsz",        str(args.imgsz),
                "--device",       args.device,
                "--project",      args.project,
                "--name",         f"blackhat_4ch_k17_blur_full_e{args.epochs}",
                "--blackhat-root","data/rdd2022_yolo/blackhat_blur",
                "--kernel-size",  "17",
                "--weights",      "yolov8s.pt",
            ],
            f"Training 4-ch BH k17+Blur — {args.epochs} epochs, batch={args.batch}, imgsz={args.imgsz}",
        )
    else:
        print("[SKIP] 4-ch training skipped.")

    # ── 3. Convergence analysis ────────────────────────────────────────────
    run(
        [
            py, "scripts/analyze_convergence_full.py",
            "--run-root", str(BASE / args.project),
            "--name-3ch", f"baseline_3ch_full_e{args.epochs}",
            "--name-4ch", f"blackhat_4ch_k17_blur_full_e{args.epochs}",
        ],
        "Running convergence analysis",
    )

    print("\n[DONE] Full comparison pipeline complete.")
    print(f"  Results:  {BASE / args.project}")
    print(f"  Analysis: {BASE / 'runs' / 'analysis_full'}")


if __name__ == "__main__":
    main()

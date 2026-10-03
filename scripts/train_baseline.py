import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=Path, default=Path("configs/rdd2022_3ch.yaml"))
    parser.add_argument("--epochs",  type=int,  default=100)
    parser.add_argument("--batch",   type=int,  default=8)
    parser.add_argument("--imgsz",   type=int,  default=640)
    parser.add_argument("--device",  type=str,  default="0")
    parser.add_argument("--project", type=str,  default="runs/rdd2022_full")
    parser.add_argument("--name",    type=str,  default="baseline_3ch_full_e100")
    args = parser.parse_args()

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

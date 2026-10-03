import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {
    "D00": 0,  # pothole
    "D10": 1,  # longitudinal crack
    "D20": 2,  # transverse crack
    "D40": 3,  # alligator cracking
}


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_MAP:
            continue
        bbox = obj.find("bndbox")
        xmin = int(bbox.findtext("xmin"))
        ymin = int(bbox.findtext("ymin"))
        xmax = int(bbox.findtext("xmax"))
        ymax = int(bbox.findtext("ymax"))
        objects.append((name, xmin, ymin, xmax, ymax))

    return filename, width, height, objects


def yolo_box(xmin, ymin, xmax, ymax, width, height):
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return x_center, y_center, box_w, box_h


def write_labels(label_path: Path, objects, width, height):
    lines = []
    for name, xmin, ymin, xmax, ymax in objects:
        cls_id = CLASS_MAP[name]
        x_center, y_center, box_w, box_h = yolo_box(
            xmin, ymin, xmax, ymax, width, height
        )
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def copy_pair(image_path: Path, label_path: Path, out_img: Path, out_lbl: Path):
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_img)
    shutil.copy2(label_path, out_lbl)


def create_split(items, seed, train_ratio, val_ratio):
    random.seed(seed)
    items = list(items)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def collect_items(input_root: Path, label_tmp: Path, prefix: str | None):
    images_dir = input_root / "images"
    xml_dir = input_root / "annotations" / "xmls"

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found in {xml_dir}")

    parsed = []
    for xml_path in xml_files:
        filename, width, height, objects = parse_xml(xml_path)
        if not filename:
            continue
        image_path = images_dir / filename
        if not image_path.exists():
            continue

        if prefix:
            new_stem = f"{prefix}_{image_path.stem}"
            new_name = f"{new_stem}{image_path.suffix}"
        else:
            new_stem = image_path.stem
            new_name = image_path.name

        label_path = label_tmp / (new_stem + ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        write_labels(label_path, objects, width, height)
        parsed.append((image_path, label_path, new_name, new_stem))

    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more RDD2022 subset train dirs (e.g., China_Drone/train)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/rdd2022_yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--debug-count", type=int, default=100)
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Do not prefix filenames with subset name",
    )
    args = parser.parse_args()

    output_root = args.output_root
    images_out = output_root / "images"
    labels_out = output_root / "labels"

    parsed = []
    label_tmp = labels_out / "tmp"
    for input_root in args.input_root:
        prefix = None if args.no_prefix else input_root.parent.name
        parsed.extend(collect_items(input_root, label_tmp, prefix))

    train_items, val_items, test_items = create_split(
        parsed, args.seed, args.train_ratio, args.val_ratio
    )

    for split, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        for image_path, label_path, new_name, new_stem in items:
            out_img = images_out / split / new_name
            out_lbl = labels_out / split / (new_stem + ".txt")
            copy_pair(image_path, label_path, out_img, out_lbl)

    debug_items = train_items[:max(0, args.debug_count)]
    for image_path, label_path, new_name, new_stem in debug_items:
        out_img = images_out / "debug" / new_name
        out_lbl = labels_out / "debug" / (new_stem + ".txt")
        copy_pair(image_path, label_path, out_img, out_lbl)

    # Cleanup temp labels
    tmp_dir = labels_out / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"Converted {len(parsed)} samples")
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()

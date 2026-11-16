"""
Dataset Preparation and Preprocessing Pipeline for YOLO Fine-Tuning.

Converts Pascal VOC XML annotations from the WT Blade Defect Dataset into standard
YOLO detection format, standardizes image resolutions to 1024x1024, partitions data
according to the established train/val/test split, and outputs data.yaml + validation metrics.
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from aeromind.vision.config import (
    BLADE_DEFECT_CLASSES,
    CLASS_TO_ID,
    PROCESSED_YOLO_DIR,
    RAW_CV_DATASET_DIR,
    STANDARD_IMAGE_SIZE,
)


def load_split_definitions(split_file: Path) -> Dict[str, str]:
    """
    Loads train/val/test mapping from train_val_test_split.txt.

    Returns:
        Dict mapping image filename (e.g. '0.jpg') to subset ('train', 'val', 'test').
    """
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    splits = {}
    with open(split_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Skip header if present (e.g. "ImageID,Subset")
    start_idx = 1 if lines[0].lower().startswith("imageid") else 0
    for line in lines[start_idx:]:
        parts = line.split(",")
        if len(parts) == 2:
            fname, subset = parts[0].strip(), parts[1].strip().lower()
            splits[fname] = subset

    return splits


def convert_voc_xml_to_yolo(
    xml_path: Path,
    class_to_id: Dict[str, int],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Parses a Pascal VOC XML annotation file and converts bounding boxes to YOLO normalized coordinates.

    Format: <class_id> <x_center> <y_center> <width> <height>
    Coordinates are normalized to [0.0, 1.0].
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is None:
        raise ValueError(f"Missing <size> tag in {xml_path}")

    img_w = float(size_elem.find("width").text)
    img_h = float(size_elem.find("height").text)

    yolo_lines: List[str] = []
    box_records: List[Dict[str, Any]] = []

    for obj in root.findall("object"):
        cname = obj.find("name").text.strip()
        if cname not in class_to_id:
            raise KeyError(f"Unknown class '{cname}' in {xml_path}. Expected one of {list(class_to_id.keys())}")

        class_id = class_to_id[cname]
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Ensure valid coordinates
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)

        # Compute normalized YOLO coordinates
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # Clamp to [0.0, 1.0]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        box_records.append({
            "class_name": cname,
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        })

    return yolo_lines, box_records


def preprocess_and_prepare_yolo_dataset(
    raw_dir: Path = RAW_CV_DATASET_DIR,
    output_dir: Path = PROCESSED_YOLO_DIR,
    target_size: Tuple[int, int] = STANDARD_IMAGE_SIZE,
    image_quality: int = 95,
) -> Dict[str, Any]:
    """
    Executes end-to-end preprocessing, VOC-to-YOLO conversion, and train/val/test splitting.
    """
    images_src_dir = raw_dir / "JPEGImages"
    annotations_src_dir = raw_dir / "Annotations"
    split_file = raw_dir / "train_val_test_split.txt"

    if not images_src_dir.exists():
        raise FileNotFoundError(f"Source images directory not found: {images_src_dir}")
    if not annotations_src_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_src_dir}")

    # Load splits
    splits = load_split_definitions(split_file)
    print(f"[INFO] Loaded split mapping for {len(splits)} images.")

    # Create destination directory tree
    subsets = ["train", "val", "test"]
    for subset in subsets:
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    summary_stats: Dict[str, Any] = {
        "dataset_name": "WT Blade Defect Dataset (YOLO Format)",
        "target_resolution": list(target_size),
        "classes": BLADE_DEFECT_CLASSES,
        "class_mapping": CLASS_TO_ID,
        "splits": {},
        "total_images": 0,
        "total_bounding_boxes": 0,
        "class_distribution_total": Counter(),
        "resized_images_count": 0,
    }

    per_split_stats = {
        s: {
            "image_count": 0,
            "box_count": 0,
            "class_distribution": Counter(),
        }
        for s in subsets
    }

    all_raw_images = sorted([f for f in images_src_dir.glob("*.jpg")])
    print(f"[INFO] Processing {len(all_raw_images)} images into {output_dir}...")

    for img_path in all_raw_images:
        fname = img_path.name
        stem = img_path.stem
        subset = splits.get(fname, "train")

        # 1. Standardize and copy image
        dst_img_path = output_dir / "images" / subset / fname
        with Image.open(img_path) as img:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Check if resize is required
            if img.size != target_size:
                img_standardized = img.resize(target_size, resample=Image.Resampling.LANCZOS)
                img_standardized.save(dst_img_path, format="JPEG", quality=image_quality)
                summary_stats["resized_images_count"] += 1
            else:
                img.save(dst_img_path, format="JPEG", quality=image_quality)

        # 2. Convert Pascal VOC XML to YOLO TXT
        xml_path = annotations_src_dir / f"{stem}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing annotation for image: {fname}")

        yolo_lines, box_records = convert_voc_xml_to_yolo(xml_path, CLASS_TO_ID)

        # Save label file
        dst_lbl_path = output_dir / "labels" / subset / f"{stem}.txt"
        with open(dst_lbl_path, "w", encoding="utf-8") as lf:
            for line in yolo_lines:
                lf.write(f"{line}\n")

        # Update statistics
        per_split_stats[subset]["image_count"] += 1
        per_split_stats[subset]["box_count"] += len(box_records)
        for box in box_records:
            cname = box["class_name"]
            per_split_stats[subset]["class_distribution"][cname] += 1
            summary_stats["class_distribution_total"][cname] += 1
        summary_stats["total_bounding_boxes"] += len(box_records)

    summary_stats["total_images"] = len(all_raw_images)

    # Convert counters to regular dicts for JSON serialization
    summary_stats["class_distribution_total"] = dict(summary_stats["class_distribution_total"])
    for s in subsets:
        summary_stats["splits"][s] = {
            "image_count": per_split_stats[s]["image_count"],
            "box_count": per_split_stats[s]["box_count"],
            "class_distribution": dict(per_split_stats[s]["class_distribution"]),
        }

    # 3. Generate data.yaml for Ultralytics YOLO
    data_yaml_path = output_dir / "data.yaml"
    abs_output_path = output_dir.resolve().as_posix()

    yaml_content = f"""# AeroMind Ops — Blade Defect Detection Dataset (Ultralytics YOLO format)
# Auto-generated by aeromind.vision.prepare_dataset

path: {abs_output_path}
train: images/train
val: images/val
test: images/test

nc: {len(BLADE_DEFECT_CLASSES)}
names:
"""
    for idx, cname in enumerate(BLADE_DEFECT_CLASSES):
        yaml_content += f"  {idx}: {cname}\n"

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[INFO] Generated YOLO configuration: {data_yaml_path}")

    # 4. Save metadata summary
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"[INFO] Generated dataset summary report: {summary_path}")

    # 5. Execute self-check validation
    validate_yolo_dataset(output_dir)

    return summary_stats


def validate_yolo_dataset(yolo_dir: Path) -> None:
    """
    Performs integrity validation on the prepared YOLO dataset.
    Verifies 1-to-1 matching between images and labels, and ensures all bounding box coordinates
    are valid normalized floats in [0, 1].
    """
    subsets = ["train", "val", "test"]
    total_imgs = 0
    total_lbls = 0
    total_boxes = 0

    for subset in subsets:
        img_dir = yolo_dir / "images" / subset
        lbl_dir = yolo_dir / "labels" / subset

        images = sorted(list(img_dir.glob("*.jpg")))
        labels = sorted(list(lbl_dir.glob("*.txt")))

        assert len(images) == len(labels), (
            f"Image/label count mismatch in {subset}: {len(images)} images vs {len(labels)} labels"
        )

        for img_p in images:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            assert lbl_p.exists(), f"Missing label file for image: {img_p}"

            with open(lbl_p, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    assert len(parts) == 5, f"Invalid label format in {lbl_p}:{line_idx}: '{line}'"
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    assert 0 <= cls_id < len(BLADE_DEFECT_CLASSES), f"Invalid class ID {cls_id} in {lbl_p}"
                    assert 0.0 <= xc <= 1.0, f"xc out of bounds in {lbl_p}: {xc}"
                    assert 0.0 <= yc <= 1.0, f"yc out of bounds in {lbl_p}: {yc}"
                    assert 0.0 <= w <= 1.0, f"w out of bounds in {lbl_p}: {w}"
                    assert 0.0 <= h <= 1.0, f"h out of bounds in {lbl_p}: {h}"
                    total_boxes += 1

        total_imgs += len(images)
        total_lbls += len(labels)

    print(f"[SUCCESS] Dataset verification passed: {total_imgs} images, {total_lbls} labels, {total_boxes} verified bounding boxes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split WT blade defect dataset for YOLO.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_CV_DATASET_DIR,
        help="Path to raw WT blade defect dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_YOLO_DIR,
        help="Destination directory for YOLO dataset",
    )
    args = parser.parse_args()

    stats = preprocess_and_prepare_yolo_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir)
    print("\n--- Summary Statistics ---")
    print(json.dumps(stats, indent=2))

from pathlib import Path
import yaml
from PIL import Image

from aeromind.vision.config import BLADE_DEFECT_CLASSES, PROCESSED_YOLO_DIR


def test_processed_yolo_dataset_structure():
    yolo_dir = PROCESSED_YOLO_DIR
    assert yolo_dir.exists(), "processed_images directory does not exist"

    # Check data.yaml
    yaml_path = yolo_dir / "data.yaml"
    assert yaml_path.exists(), "data.yaml is missing"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["nc"] == len(BLADE_DEFECT_CLASSES)
    assert len(data_cfg["names"]) == len(BLADE_DEFECT_CLASSES)

    # Check images and labels for each split
    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        assert img_dir.exists(), f"Missing images/{split}"
        assert lbl_dir.exists(), f"Missing labels/{split}"

        imgs = list(img_dir.glob("*.jpg"))
        lbls = list(lbl_dir.glob("*.txt"))
        assert len(imgs) == len(lbls), f"Mismatch in count for split {split}: {len(imgs)} vs {len(lbls)}"
        assert len(imgs) > 0, f"Split {split} is empty"


def test_yolo_label_coordinate_validity():
    yolo_dir = PROCESSED_YOLO_DIR
    for split in ["train", "val", "test"]:
        lbl_dir = yolo_dir / "labels" / split
        for lbl_path in lbl_dir.glob("*.txt"):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    assert len(parts) == 5, f"Malformed line in {lbl_path}: {line}"
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])

                    assert 0 <= cls_id < len(BLADE_DEFECT_CLASSES)
                    assert 0.0 <= xc <= 1.0, f"xc {xc} out of bounds in {lbl_path}"
                    assert 0.0 <= yc <= 1.0, f"yc {yc} out of bounds in {lbl_path}"
                    assert 0.0 <= w <= 1.0, f"w {w} out of bounds in {lbl_path}"
                    assert 0.0 <= h <= 1.0, f"h {h} out of bounds in {lbl_path}"


def test_yolo_image_resolutions():
    yolo_dir = PROCESSED_YOLO_DIR
    # Sample check 10 images from each split
    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / "images" / split
        sample_imgs = list(img_dir.glob("*.jpg"))[:10]
        for img_p in sample_imgs:
            with Image.open(img_p) as im:
                assert im.size == (1024, 1024), f"Image {img_p} has non-standard size: {im.size}"

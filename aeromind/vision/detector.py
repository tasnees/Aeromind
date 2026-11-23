"""
Blade Defect Detector for AeroMind Ops.

Provides real-time computer vision inference on turbine / compressor blades
using the fine-tuned YOLO defect inspection model.
Maps detected bounding boxes to severity levels, descriptions, and recommended
maintenance actions for Engine Maintenance Manuals (EMM).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from aeromind.vision.config import (
    BLADE_DEFECT_CLASSES,
    CLASS_TO_ID,
    DEFECT_SEVERITY_MAPPING,
    ID_TO_CLASS,
)

DEFAULT_WEIGHTS_PATH = Path("models/checkpoints/yolo_blade/exp_yolo_blade_defect/weights/best.pt")

SEVERITY_HIERARCHY = {
    "CRITICAL": 3,
    "WARNING": 2,
    "ADVISORY": 1,
    "NORMAL": 0,
}


def _resolve_weights_path(specified_path: Optional[Union[str, Path]]) -> Path:
    candidates = [
        Path(specified_path) if specified_path else None,
        Path("models/checkpoints/yolo_blade/exp_yolo_blade_defect/weights/best.pt"),
        Path("models/checkpoints/best.pt"),
        Path("best.pt"),
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return Path("models/checkpoints/yolo_blade/exp_yolo_blade_defect/weights/best.pt")


class BladeDefectDetector:
    """
    Inference and defect assessment engine for turbofan blade borescope imagery.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        augment: bool = True,
        agnostic_nms: bool = True,
        device: str = "cpu",
    ):
        self.weights_path = _resolve_weights_path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.device = device
        self.model = None

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.weights_path}\n"
                f"Please ensure best.pt is placed in {self.weights_path.parent}"
            )

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(str(self.weights_path))
            print(f"[BladeDefectDetector] Loaded YOLO model from {self.weights_path} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(
        self,
        image_path: Union[str, Path],
        save_annotated: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Runs object detection on an input blade image and assesses defects.

        Args:
            image_path: Path to the image file.
            save_annotated: If True, saves an annotated image with bounding boxes.
            output_dir: Directory to save the annotated image (default: runs/detect/predictions).

        Returns:
            Dictionary with inspection diagnosis, detected defects, and recommended maintenance actions.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Run YOLO inference with optional TTA (Test-Time Augmentation) & Class-Agnostic NMS
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            augment=self.augment,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        detected_defects: List[Dict[str, Any]] = []
        highest_severity = "NORMAL"
        recommended_actions = set()

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = [round(coord, 2) for coord in box.xyxy[0].tolist()]

                class_name = ID_TO_CLASS.get(cls_id, result.names.get(cls_id, f"class_{cls_id}"))
                meta = DEFECT_SEVERITY_MAPPING.get(class_name, {
                    "severity": "WARNING",
                    "description": "Unclassified surface anomaly.",
                    "recommended_action": "INSPECT_MANUAL",
                })

                severity = meta["severity"]
                if SEVERITY_HIERARCHY.get(severity, 0) > SEVERITY_HIERARCHY.get(highest_severity, 0):
                    highest_severity = severity

                recommended_actions.add(meta["recommended_action"])

                detected_defects.append({
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": xyxy,
                    "severity": severity,
                    "description": meta["description"],
                    "recommended_action": meta["recommended_action"],
                })

        annotated_path = None
        if save_annotated:
            out_folder = Path(output_dir) if output_dir else Path("runs/detect/predictions")
            out_folder.mkdir(parents=True, exist_ok=True)
            annotated_path = out_folder / f"annotated_{image_path.name}"
            result.save(filename=str(annotated_path))

        status = "DEFECT_DETECTED" if detected_defects else "NO_DEFECTS_DETECTED"

        return {
            "status": status,
            "image_name": image_path.name,
            "defect_count": len(detected_defects),
            "highest_severity": highest_severity,
            "recommended_actions": list(recommended_actions) if recommended_actions else ["CONTINUE_STANDARD_OPERATION"],
            "defects": detected_defects,
            "annotated_image_path": str(annotated_path) if annotated_path else None,
        }


def run_cli():
    parser = argparse.ArgumentParser(description="AeroMind Blade Defect Detector CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS_PATH), help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default 0.30)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold (default 0.45)")
    parser.add_argument("--no-augment", action="store_true", help="Disable Test-Time Augmentation (TTA)")
    parser.add_argument("--save", action="store_true", help="Save annotated image")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    args = parser.parse_args()

    detector = BladeDefectDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        augment=not args.no_augment,
        device=args.device,
    )

    assessment = detector.detect(args.image, save_annotated=args.save)
    print("\n" + "=" * 60)
    print("AEROMIND BLADE INSPECTION DIAGNOSIS")
    print("=" * 60)
    print(json.dumps(assessment, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    run_cli()

"""
Fine-tuning pipeline for YOLO on the AeroMind Blade Defect Inspection Dataset.

Supports Ultralytics YOLOv8 / YOLOv11 models, experiment tracking, and checkpointing.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from aeromind.mlops.tracker import ExperimentTracker
from aeromind.vision.config import PROCESSED_YOLO_DIR


def train_yolo_defect_model(
    model_name: str = "yolov8n.pt",
    data_yaml: Path = PROCESSED_YOLO_DIR / "data.yaml",
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 8,
    device: str = "0",
    project: str = "models/checkpoints/yolo_blade",
    name: str = "exp_yolo_blade_defect",
    use_tracker: bool = True,
) -> Any:
    """
    Fine-tunes a pretrained YOLO model on the blade defect dataset.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("\n[ERROR] Ultralytics is not installed.")
        print("Please install it with: pip install ultralytics\n")
        sys.exit(1)

    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Run aeromind.vision.prepare_dataset first.")

    tracker = None
    if use_tracker:
        tracker = ExperimentTracker(experiment_name="AeroMind_Vision_Blade_Inspection")
        tracker.start_run(run_name=f"{name}_{model_name.replace('.pt', '')}")
        tracker.log_params({
            "model_architecture": model_name,
            "data_yaml": str(data_yaml_path),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch,
            "device": device,
        })

    print(f"\n[INFO] Loading pretrained model: {model_name}...")
    model = YOLO(model_name)

    print(f"[INFO] Beginning fine-tuning on {data_yaml_path} ({epochs} epochs, imgsz={imgsz}, batch={batch})...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # Validate model on test split
    print("\n[INFO] Evaluating fine-tuned model on test split...")
    test_metrics = model.val(data=str(data_yaml_path), split="test")

    if tracker:
        try:
            # Extract evaluation metrics (mAP50, mAP50-95, precision, recall)
            metrics_dict = {
                "test_map50": float(test_metrics.box.map50),
                "test_map50_95": float(test_metrics.box.map),
                "test_precision": float(test_metrics.box.mp),
                "test_recall": float(test_metrics.box.mr),
            }
            tracker.log_metrics(metrics_dict)
            best_weight_path = Path(project) / name / "weights" / "best.pt"
            if best_weight_path.exists():
                tracker.log_artifact(best_weight_path)
            tracker.end_run(status="COMPLETED")
            print(f"[INFO] Logged run metrics to AeroMind Tracker: {metrics_dict}")
        except Exception as e:
            print(f"[WARNING] Could not record all metrics to tracker: {e}")
            tracker.end_run(status="COMPLETED")

    print("\n[SUCCESS] YOLO fine-tuning complete!")
    print(f"Weights saved at: {Path(project) / name / 'weights'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the AeroMind Blade Defect Dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (e.g. yolov8n.pt, yolov11n.pt)")
    parser.add_argument("--data", type=Path, default=PROCESSED_YOLO_DIR / "data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: '0' for GPU or 'cpu'")
    parser.add_argument("--project", type=str, default="models/checkpoints/yolo_blade", help="Save directory")
    parser.add_argument("--name", type=str, default="exp_yolo_blade_defect", help="Experiment run name")

    args = parser.parse_args()

    train_yolo_defect_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

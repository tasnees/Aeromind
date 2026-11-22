"""
AeroMind Computer Vision & Blade Defect Inspection Module.
"""

from aeromind.vision.config import (
    BLADE_DEFECT_CLASSES,
    CLASS_TO_ID,
    DEFECT_SEVERITY_MAPPING,
    ID_TO_CLASS,
    PROCESSED_YOLO_DIR,
    RAW_CV_DATASET_DIR,
)
from aeromind.vision.detector import BladeDefectDetector

__all__ = [
    "BLADE_DEFECT_CLASSES",
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "DEFECT_SEVERITY_MAPPING",
    "PROCESSED_YOLO_DIR",
    "RAW_CV_DATASET_DIR",
    "BladeDefectDetector",
]

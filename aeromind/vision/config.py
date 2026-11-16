"""
Configuration constants and settings for AeroMind Computer Vision and Blade Defect Inspection.
"""

from pathlib import Path
from typing import Dict, List

# Standard class definitions from WT blade defect dataset
BLADE_DEFECT_CLASSES: List[str] = [
    "craze",
    "corrosion",
    "surface_injure",
    "thunderstrike",
    "crack",
    "hide_craze",
]

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(BLADE_DEFECT_CLASSES)}
ID_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(BLADE_DEFECT_CLASSES)}

# Severity and maintenance action mappings for Engine Maintenance Manuals (EMM)
DEFECT_SEVERITY_MAPPING: Dict[str, Dict[str, str]] = {
    "craze": {
        "severity": "ADVISORY",
        "description": "Superficial fine surface crazing / micro-fissures in outer coating layer.",
        "recommended_action": "MONITOR",
    },
    "corrosion": {
        "severity": "WARNING",
        "description": "Chemical oxidation and surface material degradation.",
        "recommended_action": "SURFACE_TREATMENT",
    },
    "surface_injure": {
        "severity": "WARNING",
        "description": "Mechanical abrasion, minor gouges, or scratch damage.",
        "recommended_action": "BLEND_REPAIR",
    },
    "thunderstrike": {
        "severity": "CRITICAL",
        "description": "High-energy electrical arc discharge burn / localized delamination.",
        "recommended_action": "ULTRASONIC_NDT_REQUIRED",
    },
    "crack": {
        "severity": "CRITICAL",
        "description": "Structural fatigue crack propagating through substrate.",
        "recommended_action": "IMMEDIATE_REMOVAL_OR_REPLACEMENT",
    },
    "hide_craze": {
        "severity": "WARNING",
        "description": "Sub-surface micro-crazing / internal layer degradation.",
        "recommended_action": "MONITOR_SHORT_INTERVAL",
    },
}

# Standard dataset paths
RAW_CV_DATASET_DIR = Path("computer_vision/WT blade defect dataset")
PROCESSED_YOLO_DIR = Path("processed_images")
STANDARD_IMAGE_SIZE = (1024, 1024)

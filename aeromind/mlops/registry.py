import json
from pathlib import Path
from typing import Dict, List, Optional
import time


class ModelRegistry:
    """
    Manages versioning, metadata, and production promotion for AeroMind-RUL-Model.
    Provides automated promotion gates to elevate candidate models to the 'champion' alias.
    """

    def __init__(self, registry_file: Path = Path("models/registry.json")):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict] = self._load()

    def _load(self) -> Dict:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {"model_name": "AeroMind-RUL-Model", "versions": {}, "aliases": {}}
        return {"model_name": "AeroMind-RUL-Model", "versions": {}, "aliases": {}}

    def _save(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model_version(
        self,
        architecture: str,
        checkpoint_path: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, any],
    ) -> int:
        version_num = len(self.models["versions"]) + 1
        version_key = str(version_num)

        self.models["versions"][version_key] = {
            "version": version_num,
            "architecture": architecture,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "status": "candidate",
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        self.models["aliases"]["latest"] = version_num
        self._save()
        return version_num

    def evaluate_promotion_gate(
        self,
        version_num: int,
        rmse_threshold: float = 30.0,
        nasa_score_threshold: float = 5000.0,
        min_picp: float = 0.70,
    ) -> Dict[str, any]:
        """
        Validates model against production performance criteria:
        1. RMSE <= rmse_threshold
        2. NASA Score <= nasa_score_threshold
        3. If uncertainty bounds provided, PICP >= min_picp
        """
        version_key = str(version_num)
        if version_key not in self.models["versions"]:
            raise ValueError(f"Version {version_num} not registered.")

        v_data = self.models["versions"][version_key]
        metrics = v_data["metrics"]

        gate_checks = {
            "version": version_num,
            "architecture": v_data["architecture"],
            "rmse_pass": metrics.get("rmse", 999.0) <= rmse_threshold,
            "nasa_score_pass": metrics.get("nasa_score", 999999.0) <= nasa_score_threshold,
            "picp_pass": True,
            "promoted": False,
        }

        if "picp_80" in metrics:
            gate_checks["picp_pass"] = metrics["picp_80"] >= min_picp

        if gate_checks["rmse_pass"] and gate_checks["nasa_score_pass"] and gate_checks["picp_pass"]:
            gate_checks["promoted"] = True
            self.promote_to_champion(version_num)

        return gate_checks

    def promote_to_champion(self, version_num: int):
        version_key = str(version_num)
        if version_key not in self.models["versions"]:
            raise ValueError(f"Version {version_num} not registered.")

        # Demote previous champion
        prev_champ = self.models["aliases"].get("champion")
        if prev_champ and str(prev_champ) in self.models["versions"]:
            self.models["versions"][str(prev_champ)]["status"] = "archived"

        self.models["versions"][version_key]["status"] = "champion"
        self.models["aliases"]["champion"] = version_num
        self._save()

    def get_champion_model(self) -> Optional[Dict]:
        champ_version = self.models["aliases"].get("champion")
        if not champ_version:
            # Fallback to latest
            champ_version = self.models["aliases"].get("latest")
        if champ_version:
            return self.models["versions"].get(str(champ_version))
        return None


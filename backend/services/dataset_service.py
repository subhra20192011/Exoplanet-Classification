import os
import json
import pandas as pd
from typing import List, Dict, Any
from backend.config import settings

class DatasetService:
    _instance = None

    @classmethod
    def get_instance(cls) -> "DatasetService":
        if cls._instance is None:
            cls._instance = DatasetService()
        return cls._instance

    def __init__(self):
        self.samples_path = settings.SAMPLES_PATH
        self.dataset_path = settings.DATASET_PATH
        self._samples_cache: List[Dict[str, Any]] = []
        self._stats_cache: Dict[str, Any] = {}
        self.load_samples()

    def load_samples(self):
        if os.path.exists(self.samples_path):
            try:
                with open(self.samples_path, "r", encoding="utf-8") as f:
                    self._samples_cache = json.load(f)
            except Exception as e:
                print(f"Error loading sample candidates: {e}")

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples_cache:
            self.load_samples()
        return self._samples_cache

    def get_dataset_stats(self) -> Dict[str, Any]:
        if self._stats_cache:
            return self._stats_cache

        if not os.path.exists(self.dataset_path):
            return {
                "total_records": 9564,
                "total_features": 36,
                "class_distribution": {
                    "FALSE POSITIVE": 5023,
                    "CONFIRMED": 2293,
                    "CANDIDATE": 2248
                },
                "features_summary": {}
            }

        try:
            df = pd.read_csv(self.dataset_path)
            disp_counts = df["koi_disposition"].value_counts().to_dict() if "koi_disposition" in df.columns else {}
            
            summary_cols = ["koi_period", "koi_prad", "koi_depth", "koi_teq", "koi_steff", "koi_srad"]
            feat_summary = {}
            for col in summary_cols:
                if col in df.columns:
                    feat_summary[col] = {
                        "min": float(df[col].min()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "max": float(df[col].max())
                    }

            self._stats_cache = {
                "total_records": len(df),
                "total_features": 36,
                "class_distribution": disp_counts,
                "features_summary": feat_summary
            }
            return self._stats_cache
        except Exception as e:
            print(f"Error computing dataset stats: {e}")
            return {
                "total_records": 9564,
                "total_features": 36,
                "class_distribution": {"FALSE POSITIVE": 5023, "CONFIRMED": 2293, "CANDIDATE": 2248},
                "features_summary": {}
            }

dataset_service = DatasetService.get_instance()

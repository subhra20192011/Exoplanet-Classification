import os
from typing import Dict, Any, List
from ml.inference import KeplerInferenceEngine
from backend.config import settings

class ModelService:
    _instance = None

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def __init__(self):
        self.engine = KeplerInferenceEngine(model_path=settings.MODEL_PATH)

    def is_loaded(self) -> bool:
        return self.engine.model is not None

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.predict(input_data)

    def get_metadata(self) -> Dict[str, Any]:
        return self.engine.get_model_metadata()

model_service = ModelService.get_instance()

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnxruntime as rt
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from sneakers_ml.features.base import BaseFeatures
from sneakers_ml.models.onnx_utils import get_session, predict


class Feature(TypedDict):
    feature_instance: BaseFeatures
    idx_to_class: dict[int, str]
    model_instances: dict[str, rt.InferenceSession]


class BrandsClassifier:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.instances: dict[str, Feature] = {}
        start_time = time.time()
        logger.info("Loading models: " + ", ".join(self.config.models.keys()))

        for feature in self.config.features:
            feature_instance: BaseFeatures = instantiate(
                config=self.config.features[feature], config_data=self.config.data
            )
            class_to_idx = feature_instance.get_class_to_idx()
            self.instances[feature] = {
                "feature_instance": feature_instance,
                "idx_to_class": {ind: cls for cls, ind in class_to_idx.items()},
                "model_instances": {},
            }

            for model in self.config.models:
                model_path = Path(self.config.paths.models_save) / f"{feature}-{model}.onnx"
                self.instances[feature]["model_instances"][model] = get_session(str(model_path))
        end_time = time.time()
        logger.info(f"All models loaded in {end_time - start_time:.1f} seconds")

    def _predict_feature(
        self, feature_name: str, images: Sequence[Image.Image]
    ) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        result: dict[str, np.ndarray] = {}
        string_result: dict[str, list[str]] = {}
        embedding = self.instances[feature_name]["feature_instance"].get_features(images)
        ind_to_class = self.instances[feature_name]["idx_to_class"]
        for model_name, model in self.instances[feature_name]["model_instances"].items():
            pred = predict(model, embedding).astype(np.int32)
            result[f"{feature_name}-{model_name}"] = pred
            string_result[f"{feature_name}-{model_name}"] = [ind_to_class.get(i) for i in pred]

        return result, string_result

    def predict(self, images: Sequence[Image.Image]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        predictions: dict[str, np.ndarray] = {}
        string_predictions: dict[str, np.ndarray] = {}
        for feature_name in self.instances:
            result, string_result = self._predict_feature(feature_name, images)
            predictions |= result
            string_predictions |= string_result
        return string_predictions


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="ml-predict"):
        cfg = compose(config_name="cfg_ml")
        image = Image.open("data/training/brands-classification/train/adidas/1.jpeg")
        print(BrandsClassifier(cfg).predict([image]))
        print(BrandsClassifier(cfg).predict([image, image, image]))

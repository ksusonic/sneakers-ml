import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import onnxruntime as rt
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from sneakers_ml.features.base import BaseFeatures
from sneakers_ml.models.onnx_utils import get_session, predict
from sneakers_ml.models.resnet152_classification import Resnet152Classifier
from sneakers_ml.models.vit_classification import ViTClassifier

if TYPE_CHECKING:
    from sneakers_ml.models.base import DLClassifier


class Feature(TypedDict):
    feature_instance: BaseFeatures
    idx_to_class: dict[int, str]
    model_instances: dict[str, rt.InferenceSession]


class BrandsClassifier:
    def __init__(self, config_ml: DictConfig, config_dl: DictConfig) -> None:
        self.config_ml = config_ml
        self.config_dl = config_dl

        start_time = time.time()
        self.instances: dict[str, Feature] = {}
        if self.config_ml.get("models"):

            logger.info("Loading ML models: " + ", ".join(self.config_ml.models.keys()))

            for feature in self.config_ml.features:
                feature_instance: BaseFeatures = instantiate(
                    config=self.config_ml.features[feature], config_data=self.config_ml.data
                )
                class_to_idx = feature_instance.get_class_to_idx()
                self.instances[feature] = {
                    "feature_instance": feature_instance,
                    "idx_to_class": {ind: class_ for class_, ind in class_to_idx.items()},
                    "model_instances": {},
                }

                for model in self.config_ml.models:
                    model_path = Path(self.config_ml.paths.models_save) / f"{feature}-{model}.onnx"
                    self.instances[feature]["model_instances"][model] = get_session(str(model_path))
        else:
            logger.info("No ML models set")

        self.dl_models: dict[str, DLClassifier] = {}
        if self.config_dl.get("models"):
            if "resnet152" in self.config_dl.models:
                self.dl_models[self.config_dl.models.resnet152.name] = Resnet152Classifier(
                    self.config_dl.models.resnet152.onnx_path, self.config_dl.models.resnet152.class_to_idx
                )
            if "vit" in self.config_dl.models:
                self.dl_models[self.config_dl.models.vit.name] = ViTClassifier(
                    self.config_dl.models.vit.onnx_path,
                    self.config_dl.models.vit.class_to_idx,
                    self.config_dl.models.vit.hf_name,
                )
            logger.info("Loaded dl models: " + ", ".join(self.dl_models.keys()))
        else:
            logger.info("No DL models set")

        end_time = time.time()
        logger.info(f"All models loaded in {end_time - start_time:.1f} seconds")

    def _predict_feature(self, feature_name: str, images: Sequence[Image.Image]) -> dict[str, list[str]]:
        string_result: dict[str, list[str]] = {}
        embedding = self.instances[feature_name]["feature_instance"].get_features(images)
        idx_to_class = self.instances[feature_name]["idx_to_class"]
        for model_name, model in self.instances[feature_name]["model_instances"].items():
            pred = predict(model, embedding).astype(np.int32)
            string_result[f"{feature_name}-{model_name}"] = [idx_to_class[i] for i in pred]
        return string_result

    def predict(self, images: Sequence[Image.Image]) -> dict[str, list[str]]:
        string_predictions: dict[str, list[str]] = {}
        for feature_name in self.instances:
            string_result = self._predict_feature(feature_name, images)
            string_predictions |= string_result

        for model_name, model in self.dl_models.items():
            pred = model.predict(images)
            string_predictions[model_name] = pred

        return string_predictions


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="ml-predict"):
        cfg_ml = compose(config_name="cfg_ml")
    with initialize(version_base=None, config_path="../../config", job_name="ml-predict"):
        cfg_dl = compose(config_name="cfg_dl")
    image = Image.open("tests/static/newbalance574.jpg")
    print(BrandsClassifier(cfg_ml, cfg_dl).predict([image]))
    print(BrandsClassifier(cfg_ml, cfg_dl).predict([image, image, image]))

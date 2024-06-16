from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from hydra import compose, initialize
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from sneakers_ml.models.base import SimilaritySearchBase, SimilaritySearchPredictor, SimilaritySearchTrainer
from sneakers_ml.models.onnx_utils import (
    predict_clip_image,
    predict_clip_text,
    save_clip_text_model,
    save_clip_vision_model,
)
from sneakers_ml.models.qdrant import Qdrant


class CLIPSimilaritySearchTrainer(SimilaritySearchTrainer):
    def __init__(
        self,
        image_folder: str,
        clip_model_name: str,
        onnx_path: str,
        onnx_path_vision: str,
        embeddings_path: str,
        device: str,
        qdrant: Optional[Qdrant] = None,
    ) -> None:
        super().__init__(
            image_folder=image_folder,
            embeddings_path=embeddings_path,
            onnx_path=onnx_path,
            device=device,
            qdrant=qdrant,
        )
        self.onnx_path_vision = onnx_path_vision
        self.clip_model_name = clip_model_name
        self.processor: CLIPProcessor = None
        self.clip_model: CLIPModel = None

    def init_data(self) -> None:
        self.dataset = ImageFolder(self.image_folder, transform=lambda x: self.processor(images=x, return_tensors="pt"))
        self.dataloader = DataLoader(
            self.dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=6, pin_memory=False
        )

    def init_model(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_model.eval()

    def create_onnx_model(self) -> None:
        model = CLIPTextModelWithProjection.from_pretrained(self.clip_model_name)
        model.eval()
        inputs = self.processor(text=["a dummy sentence"], return_tensors="pt", padding=True)
        save_clip_text_model(model, tuple(inputs.values()), self.onnx_path)

        model = CLIPVisionModelWithProjection.from_pretrained(self.clip_model_name)
        model.eval()
        inputs = self.processor(images=torch.rand(1, 3, 224, 224), return_tensors="pt")
        save_clip_vision_model(model, tuple(inputs.values()), self.onnx_path_vision)

    def model_forward(self, data: Sequence[Any]) -> torch.Tensor:
        images = data[0].to(self.device)["pixel_values"].squeeze(1)
        return self.clip_model.get_image_features(pixel_values=images).cpu()  # type: ignore[no-any-return]


class CLIPTextToImageSimilaritySearch(SimilaritySearchPredictor):
    def __init__(
        self,
        embeddings_path: str,
        onnx_path: str,
        metadata_path: str,
        clip_model_name: str,
        qdrant: Optional[Qdrant] = None,
    ) -> None:
        super().__init__(embeddings_path, onnx_path, metadata_path, qdrant=qdrant)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def get_features(self, text_query: Optional[Union[Sequence[str], str]] = None) -> np.ndarray:
        inputs = self.processor(text=text_query, return_tensors="np", padding=True, truncation=True)
        return predict_clip_text(self.onnx_session, inputs)

    def predict(self, top_k: int, text_query: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
        return self.get_similar(self.get_features(text_query), top_k)


class CLIPSimilaritySearch(SimilaritySearchPredictor):
    def __init__(
        self,
        embeddings_path: str,
        onnx_path: str,
        metadata_path: str,
        clip_model_name: str,
        qdrant: Optional[Qdrant] = None,
    ) -> None:
        super().__init__(embeddings_path, onnx_path, metadata_path, qdrant=qdrant)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def get_features(self, images: Optional[Union[Sequence[Image.Image], Image.Image]] = None) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="np")
        return predict_clip_image(self.onnx_session, inputs)

    def predict(self, top_k: int, image: Optional[Image.Image] = None) -> tuple[np.ndarray, np.ndarray]:
        return self.get_similar(self.get_features(image), top_k)


if __name__ == "__main__":
    # train
    with initialize(version_base=None, config_path="../../config", job_name="clip-text2image-create"):
        cfg = compose(config_name="cfg_clip")
        trainer = CLIPSimilaritySearchTrainer(
            cfg.images_path, cfg.base_model, cfg.text_model_path, cfg.vision_model_path, cfg.embeddings_path, cfg.device
        )
        trainer.train()

    # predict text
    with initialize(version_base=None, config_path="../../config", job_name="clip-text2image-predict"):
        cfg = compose(config_name="cfg_clip")
        predictor = CLIPTextToImageSimilaritySearch(
            cfg.embeddings_path, cfg.text_model_path, cfg.metadata_path, cfg.base_model
        )
        print(predictor.predict(3, "blue sneakers"))

    # predict image
    with initialize(version_base=None, config_path="../../config", job_name="clip-similarity-search-predict"):
        cfg = compose(config_name="cfg_clip")
        predictor = CLIPSimilaritySearch(cfg.embeddings_path, cfg.vision_model_path, cfg.metadata_path, cfg.base_model)
        test_image = Image.open("tests/static/newbalance574.jpg")
        print(predictor.predict(3, test_image))

    # create qdrant vectors
    with initialize(version_base=None, config_path="../../config", job_name="qdrant-clip-create"):
        cfg = compose(config_name="cfg_clip")
        test_qdrant = Qdrant(cfg.qdrant_host, cfg.qdrant_port, cfg.qdrant_collection_name)
        numpy_features, classes, class_to_idx = SimilaritySearchBase.load_features(cfg.embeddings_path)
        test_qdrant.save_features(numpy_features, classes, class_to_idx)

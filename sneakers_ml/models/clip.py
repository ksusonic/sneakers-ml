from collections.abc import Sequence
from typing import Any
from typing import Union

import numpy as np
import torch
from hydra import compose
from hydra import initialize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import CLIPModel
from transformers import CLIPProcessor
from transformers import CLIPTextModelWithProjection

from sneakers_ml.models.onnx_utils import predict_clip
from sneakers_ml.models.onnx_utils import save_clip_model
from sneakers_ml.models.similarity_search_base import SimilaritySearchPredictor
from sneakers_ml.models.similarity_search_base import SimilaritySearchTrainer


class CLIPSimilaritySearchTrainer(SimilaritySearchTrainer):
    """ """

    def __init__(self, image_folder: str, clip_model_name: str, onnx_path: str,
                 embeddings_path: str, device: str) -> None:
        super().__init__(image_folder=image_folder,
                         embeddings_path=embeddings_path,
                         onnx_path=onnx_path,
                         device=device)

        self.clip_model_name = clip_model_name
        self.processor = None
        self.clip_model = None

    def init_data(self) -> None:
        """ """
        self.dataset = ImageFolder(
            self.image_folder,
            transform=lambda x: self.processor(images=x, return_tensors="pt"))
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=128,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=6,
                                     pin_memory=False)

    def init_model(self) -> None:
        """ """
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(
            self.device)
        self.clip_model.eval()

    def create_onnx_model(self) -> None:
        """ """
        model = CLIPTextModelWithProjection.from_pretrained(
            self.clip_model_name)
        model.eval()

        text = ["a dummy sentence"]
        inputs = self.processor(text=text, return_tensors="pt", padding=True)

        save_clip_model(model, tuple(inputs.values()), self.onnx_path)

    def model_forward(self, data: Sequence[Any]) -> torch.Tensor:
        """

        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:
        :param data: Sequence[Any]:

        """
        images = data[0].to(self.device)["pixel_values"].squeeze(1)
        return self.clip_model.get_image_features(pixel_values=images).cpu()


class CLIPTextToImageSimilaritySearch(SimilaritySearchPredictor):
    """ """

    def __init__(self, embeddings_path: str, onnx_path: str,
                 metadata_path: str, clip_model_name: str) -> None:
        super().__init__(embeddings_path, onnx_path, metadata_path)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def get_features(self,
                     text_query: Union[Sequence[str],
                                       str] = None) -> np.ndarray:
        """

        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str: Default value = None)
        :param text_query: Union[Sequence[str]:
        :param str]:  (Default value = None)

        """
        inputs = self.processor(text=text_query,
                                return_tensors="np",
                                padding=True)
        return predict_clip(self.onnx_session, inputs)

    def predict(self,
                top_k: int,
                text_query: str = None) -> tuple[np.ndarray, np.ndarray]:
        """

        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)
        :param top_k: int:
        :param text_query: str:  (Default value = None)

        """
        return self.get_similar(self.get_features(text_query), top_k)


if __name__ == "__main__":
    # train
    with initialize(version_base=None,
                    config_path="../../config",
                    job_name="text2image-features-create"):
        cfg = compose(config_name="cfg_text_to_image")
        trainer = CLIPSimilaritySearchTrainer(cfg.images_path, cfg.base_model,
                                              cfg.model_path,
                                              cfg.embeddings_path, cfg.device)
        trainer.train()

    # predict
    with initialize(version_base=None,
                    config_path="../../config",
                    job_name="text2image-features-predict"):
        cfg = compose(config_name="cfg_text_to_image")
        predictor = CLIPTextToImageSimilaritySearch(cfg.embeddings_path,
                                                    cfg.model_path,
                                                    cfg.metadata_path,
                                                    cfg.base_model)
        print(predictor.predict(3, "blue sneakers"))

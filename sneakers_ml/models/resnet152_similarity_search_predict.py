from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import ResNet152_Weights

from sneakers_ml.models.onnx_utils import get_device, get_session, predict
from sneakers_ml.models.resnet152_similarity_search_train import ResNet152SimilaritySearchCreator


class ResNet152SimilaritySearch:
    def __init__(self, embeddings_path: str, onnx_path: str, metadata_path: str, device: str) -> None:
        self.embeddings_path = embeddings_path
        self.onnx_path = onnx_path
        self.device = get_device(device)
        self.metadata_path = metadata_path
        self.numpy_features, self.classes, self.class_to_idx = ResNet152SimilaritySearchCreator._load_features(
            self.embeddings_path
        )

        self.idx_to_class = {str(v): k for k, v in self.class_to_idx.items()}

        self.weights = ResNet152_Weights.DEFAULT
        self.preprocess = self.weights.transforms()

        self.onnx_session = get_session(self.onnx_path, self.device)

        self.df = pd.read_csv(self.metadata_path)
        self.df = self.df.drop(
            ["brand_merge", "images_path", "collection_name", "color", "images_flattened", "title_without_color"],
            axis=1,
        )
        self.df["title"] = self.df["title"].apply(eval)
        self.df["brand"] = self.df["brand"].apply(eval)
        self.df["price"] = self.df["price"].apply(eval)
        self.df["pricecurrency"] = self.df["pricecurrency"].apply(eval)
        self.df["website"] = self.df["website"].apply(eval)
        self.df["url"] = self.df["url"].apply(eval)
        self.df = self.df.explode(["title", "brand", "price", "pricecurrency", "url", "website"])

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image)  # type: ignore[no-any-return]

    def _get_feature(self, image: Image.Image) -> np.ndarray:
        return self.get_features([image])

    def get_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        preprocessed_images = torch.stack([self.apply_transforms(image) for image in images])
        return predict(self.onnx_session, preprocessed_images)

    def get_similar(self, image: Image.Image, threshold_low: float, threshold_high: float):
        image_feature = self._get_feature(image)
        similarity_matrix = cosine_similarity(self.numpy_features, image_feature).flatten()
        similar_indices = np.argwhere(
            (similarity_matrix >= threshold_low) & (similarity_matrix <= threshold_high)
        ).flatten()

        similar_objects = self.classes[similar_indices]
        # similar_images = similar_objects[:, 0]
        similar_models = np.vectorize(self.idx_to_class.get)(similar_objects[:, 1])

        similar_metadata_dump = (
            self.df[self.df["title_merge"].isin(set(similar_models.tolist()))]
            .groupby(["title", "website"])
            .agg(
                {
                    "title_merge": "first",
                    "brand": "first",
                    "price": lambda x: f"{min(x)} - {max(x)}",
                    "pricecurrency": "first",
                    "url": "first",
                }
            )
            .reset_index()
            .to_numpy()
        )
        return similar_metadata_dump


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config", job_name="similarity-search-features-predict"):
        cfg = compose(config_name="cfg_similarity_search")

        temp = ResNet152SimilaritySearch(cfg.embeddings_path, cfg.model_path, cfg.metadata_path, cfg.device)

        image = Image.open("data/training/brands-classification/train/adidas/1.jpeg")
        print(temp.get_similar(image, 0.85, 0.9))

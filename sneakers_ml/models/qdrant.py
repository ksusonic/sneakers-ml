import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class Qdrant:
    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self.host = host
        self.port = port
        self.client = QdrantClient(host=self.host, port=self.port)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def save_features(self, numpy_features: np.ndarray, classes: np.ndarray, class_to_idx: dict[str, int]) -> None:

        self.create_collection(vector_size=numpy_features.shape[1])

        idx_to_class = {str(v): k for k, v in class_to_idx.items()}
        points = []
        for i, row in enumerate(classes):
            points.append(
                PointStruct(
                    id=i,
                    vector=numpy_features[i].tolist(),
                    payload={
                        "image_path": row[0],
                        "class": idx_to_class[row[1]],
                    },
                )
            )
        self.client.upload_points(collection_name=self.collection_name, points=points, parallel=4, max_retries=3)

    def get_similar(self, feature: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        similar_objects = self.client.search(
            collection_name=self.collection_name, query_vector=feature[0].tolist(), limit=top_k
        )
        similar_images = np.array([item.payload["image_path"] for item in similar_objects])
        similar_models = np.array([item.payload["class"] for item in similar_objects])

        return similar_images, similar_models

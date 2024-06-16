import os

from hydra import compose, initialize
from loguru import logger

from sneakers_ml.models.base import SimilaritySearchBase
from sneakers_ml.models.qdrant import Qdrant

QDRANT_CLIENT = None

with initialize(version_base=None, config_path="../../../../config", job_name="fastapi"):
    cfg = compose(config_name="cfg_clip")
    QDRANT_CLIENT = Qdrant(os.getenv("QDRANT_HOST") or cfg.qdrant_host, cfg.qdrant_port, cfg.qdrant_collection_name)
    numpy_features, classes, class_to_idx = SimilaritySearchBase.load_features(cfg.embeddings_path)
    QDRANT_CLIENT.save_features(numpy_features, classes, class_to_idx)
    logger.info("loaded Qdrant client")

import os
import time
from hashlib import md5

from fastapi import APIRouter, UploadFile
from hydra import compose, initialize
from loguru import logger
from PIL import Image

from sneakers_ml.app.config import config
from sneakers_ml.app.models.image import Response
from sneakers_ml.app.service.qdrant import QDRANT_CLIENT
from sneakers_ml.app.service.redis import RedisCache
from sneakers_ml.models.clip import CLIPSimilaritySearch

searcher: CLIPSimilaritySearch = None
redis = RedisCache(host=config.redis_host, port=config.redis_port)

router: APIRouter = APIRouter(prefix="/similarity-search", tags=["similarity-search"])


@router.on_event("startup")
async def load_model():
    configs_rel_path = os.path.relpath(config.ml_config_path, start=os.path.dirname(os.path.abspath(__file__)))
    with initialize(version_base=None, config_path=str(configs_rel_path), job_name="fastapi"):
        cfg = compose(config_name="cfg_clip")

        global searcher
        searcher = CLIPSimilaritySearch(
            cfg.embeddings_path, cfg.vision_model_path, cfg.metadata_path, cfg.base_model, QDRANT_CLIENT
        )
    logger.info("Loaded CLIPSimilaritySearch")


@router.post("/upload/")
async def post_similarity_search(image: UploadFile) -> Response:
    image.file.seek(0)
    image = Image.open(image.file)

    image_key = md5(image.tobytes()).hexdigest()  # nosec B324 (weak hash)
    cached_search = redis.get(image_key)
    if cached_search:
        logger.info("Found cached search for image: {}", image_key)
        return cached_search

    start_time = time.time()
    predictions = searcher.predict(3, image)
    end_time = time.time()
    logger.info("Predicted {} in {} seconds", predictions, end_time - start_time)

    redis.set(image_key, predictions, ttl=3600)

    metadata = list(
        map(
            lambda metadata: Response.Metadata(
                title=metadata[0],
                brand=metadata[3],
                dataset=metadata[1],
                price=f"{metadata[4]} {metadata[5]}",
                url=metadata[6],
            ),
            predictions[0].tolist(),
        )
    )

    images = list(
        map(lambda image: f"https://storage.yandexcloud.net/sneaker-ml-models/{image}", predictions[1].tolist())
    )

    return Response(
        images=images,
        metadata=metadata,
    )

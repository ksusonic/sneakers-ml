import os
import time

from fastapi import APIRouter
from hydra import compose, initialize
from loguru import logger

from sneakers_ml.app.config import config
from sneakers_ml.app.models.image import Request, Response
from sneakers_ml.app.service.redis import RedisCache
from sneakers_ml.models.clip import CLIPTextToImageSimilaritySearch

searcher: CLIPTextToImageSimilaritySearch = None
redis = RedisCache(host=config.redis_host, port=config.redis_port)

router: APIRouter = APIRouter(prefix="/text-to-image", tags=["text-to-image"])


@router.on_event("startup")
async def load_model():
    configs_rel_path = os.path.relpath(config.ml_config_path, start=os.path.dirname(os.path.abspath(__file__)))
    with initialize(version_base=None, config_path=str(configs_rel_path), job_name="fastapi"):
        cfg = compose(config_name="cfg_text_to_image")
        global searcher
        searcher = CLIPTextToImageSimilaritySearch(
            cfg.embeddings_path, cfg.model_path, cfg.metadata_path, cfg.base_model
        )
    logger.info("Loaded CLIPTextToImageSimilaritySearch")


@router.post("/clip/")
async def post_text_to_image(request: Request) -> Response:
    cached_prediction = redis.get(request.text)
    if cached_prediction:
        logger.info("Found cached prediction for image: {}", request.text)
        return cached_prediction

    start_time = time.time()
    predictions = searcher.predict(top_k=3, text_query=request.text)
    end_time = time.time()
    logger.info("Predicted {} in {} seconds", predictions, end_time - start_time)

    redis.set(request.text, predictions, ttl=3600)

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

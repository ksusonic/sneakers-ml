import os
import time
from hashlib import md5

from fastapi import APIRouter, UploadFile
from hydra import compose, initialize
from loguru import logger
from PIL import Image

from sneakers_ml.app.config import config
from sneakers_ml.app.service.redis import RedisCache
from sneakers_ml.models.predict import BrandsClassifier

predictor: BrandsClassifier = None
redis = RedisCache(host=config.redis_host, port=config.redis_port)

router: APIRouter = APIRouter(prefix="/classify-brand", tags=["brand-classification"])


@router.on_event("startup")
async def load_model():
    configs_rel_path = os.path.relpath(config.ml_config_path, start=os.path.dirname(os.path.abspath(__file__)))
    with initialize(version_base=None, config_path=str(configs_rel_path), job_name="fastapi"):
        cfg_ml = compose(config_name="cfg_ml")
        cfg_dl = compose(config_name="cfg_dl")
        global predictor
        predictor = BrandsClassifier(cfg_ml, cfg_dl)
    logger.info("Loaded BrandsClassifier")


@router.post("/upload/")
async def post_image_to_classify(image: UploadFile):
    image.file.seek(0)
    image = Image.open(image.file)

    image_key = md5(image.tobytes()).hexdigest()  # nosec B324 (weak hash)
    cached_prediction = redis.get(image_key)
    if cached_prediction:
        logger.info("Found cached prediction for image: {}", image_key)
        return cached_prediction

    start_time = time.time()
    predictions = predictor.predict(images=[image])[1]
    end_time = time.time()
    logger.info("Predicted {} in {} seconds", predictions, end_time - start_time)

    redis.set(image_key, predictions, ttl=3600)
    return predictions

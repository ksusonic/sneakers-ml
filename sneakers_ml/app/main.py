import os

from fastapi import FastAPI
from loguru import logger

from sneakers_ml.app.routers import classification, common, text_to_image
from sneakers_ml.app.version import __version__

app = FastAPI(
    title="Sneakers ML backend",
    description="This backend predicts brand of sneakers by image.",
    version=__version__,
    debug=os.getenv("DEBUG", False),
)

# Add Routers
app.include_router(common.router)
app.include_router(classification.router)
app.include_router(text_to_image.router)

logger.info("App started")

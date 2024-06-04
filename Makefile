ifneq (,$(wildcard ./.env))
    include .env
    export
endif

define download_models
if [ -d "data" ]; then \
   echo "Models are downloaded"; \
else \
	echo "Pulling latest models..."; \
	wget -nc "https://sneaker-ml-models.website.yandexcloud.net/brand-cls-models.tar.gz"; \
	tar -xvf brand-cls-models.tar.gz; \
fi
endef

bot:
	poetry install --with bot
	python sneakers_ml/bot/main.py

api:
	poetry install --with api
	echo "Pulling latest dvc data..."
	dvc pull data/models/brands-classification*.dvc data/features/brands-classification*.dvc -f
	uvicorn sneakers_ml.app.main:app --proxy-headers --host 0.0.0.0 --port 8000

load-models:
	$(call download_models)

test:
	poetry install --with test
	pytest sneakers_ml tests

compress-models:
	tar -czf build/brand-cls-models.tar.gz -C ./data/models/brands-classification/ . -C ./data/features/brands-classification/ .

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

bot:
	poetry install --with bot
	python sneakers_ml/bot/main.py

api:
	poetry install --with api
	echo "Pulling latest dvc data..."
	dvc pull data/models/brands-classification*.dvc data/features/brands-classification*.dvc -f
	uvicorn sneakers_ml.app.main:app --proxy-headers --host 0.0.0.0 --port 8000

test:
	poetry install --with test
	pytest sneakers_ml tests

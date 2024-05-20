ifneq (,$(wildcard ./.env))
    include .env
    export
endif

bot:
	python sneakers_ml/bot/main.py

app_local:
	uvicorn sneakers_ml.app.main:app --reload --access-log --host localhost --port 8000

app:
	echo "Pulling latest dvc data..."
	dvc pull data/models/brands-classification.dvc data/features/brands-classification.dvc -f
	uvicorn sneakers_ml.app.main:app --proxy-headers --host 0.0.0.0 --port 8000

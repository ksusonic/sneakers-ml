ifneq (,$(wildcard ./.env))
    include .env
    export
endif

bot:
	python sneakers_ml/bot/main.py

app:
	echo "Pulling latest dvc data..."
	dvc pull data/models/brands-classification*.dvc data/features/brands-classification*.dvc -f
	uvicorn sneakers_ml.app.main:app --proxy-headers --host localhost --port 8000

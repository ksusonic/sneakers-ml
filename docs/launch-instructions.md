# Инструкции по запуску

## Общее

Проект использует poetry в качестве менеджера библиотек Python.

```shell
poetry install --with api,bot,data-eda,dev
```

Проект использует dvc для хранения данных и моделей

```shell
dvc pull
```

Конфигурация производится с помощью Hydra. Она хранится в папке [config](/config) и используется по всему проекту

## FastAPI и телеграм бот

### Docker

В репозитории настроен per-commit flow в Github Actions, в котором собираются образы Api и Bot контейнеров.
На настоящее время результат выгружается в Yandex Container Registry (до этого в Github Registry)

| Api                                                   | Bot                                                   |
| ----------------------------------------------------- | ----------------------------------------------------- |
| cr.yandex/crp9sd2f3p1o3mfu9664/sneakers-ml-api:latest | cr.yandex/crp9sd2f3p1o3mfu9664/sneakers-ml-bot:latest |

Для простого поднятия сервисов создана docker-compose конфигурация:

- Local Docker-compose для локальной сборки: [docker-compose.local.yml](/docker-compose.local.yml)
- Docker-compose для production-развертывания: [docker-compose.yml](/docker-compose.yml)

Все переменные окружения должны браться из .env файлов (путем создания копии эталона [.env.dist](/.env.dist))

### FastAPI

```shell
bash run-app.sh
```

### Telegram bot

```shell
BOT_TOKEN=XXX python sneakers_ml/bot/main.py
```

## Данные и ML

Для парсинга достаточно запустить файл соответствующий сайту

```shell
python sneakers_ml/data/parser/sneakerbaas.py
```

Объединение данных

```shell
python sneakers_ml/data/merger/merger.py
```

Генерация признаков всеми используемыми методами

```shell
python sneakers_ml/features/base.py
```

Обучение моделей

```shell
python sneakers_ml/models/train.py
```

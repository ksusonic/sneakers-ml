# Структура проекта

- Данные, эмбеддинги и модели храним с помощью DVC на s3 - [link](https://console.cloud.yandex.ru/folders/b1gkpsnq6bd5s58dgqre/storage/buckets/sneakers-ml)
- Используем poetry для установки Python библиотек
- Добавили различные линтеры и форматтеры
- Настроили pre-commit
- Модели храним в onnx, pickle не используем
- Настроили конфигурацию с помощью Hydra
- Реализовали телеграмм-бот, CI/CD Docker-образа в Registry
- Документация проекта в .md файлах в /docs
- Основной код в .py файлах, ноутбуки используем только как черновик
- Метрики логируем в WandB

## Описание директорий

```tree
sneakers-ml
├── config                - файлы конфигурации Hydra
├── data                  - папка на dvc
│   ├── features          - папка с эмбеддингами картинок
│   ├── merged            - папка с объединёнными датасетами
│   ├── models            - папка с сохранёнными моделями
│   ├── raw               - папка со спаршенными данными
│   └── training          - папка с преобразованными для тренировки данными, сплиты на тренировку и валидацию
├── deploy                - деплой сервиса
│   ├── app               - FastAPI DockerFile
│   └── bot               - TelegramBot DockerFile
├── docker-compose.yml
├── docs                  - документацичя проекта
├── LICENSE
├── Makefile
├── notebooks             - ноутбуки для исследования и первичного написания кода (черновик)
├── poetry.lock
├── pyproject.toml
├── README.md
└── sneakers_ml
    ├── app               - код FastAPI
    ├── bot               - код Telegram Bot
    ├── data              - код для парсинга, чистки и объединения данных
    ├── features          - код для генерации эмбеддингов
    └── models            - код для обучения моделей
```

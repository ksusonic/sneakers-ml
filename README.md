# Платформа для поиска похожих кроссовок

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/miem-refugees/sneakers-ml/trunk.svg)](https://results.pre-commit.ci/latest/github/miem-refugees/sneakers-ml/trunk)
[![API docker](https://github.com/miem-refugees/sneakers-ml/actions/workflows/build-api.yml/badge.svg)](https://github.com/miem-refugees/sneakers-ml/actions/workflows/build-api.yml)
[![BOT docker](https://github.com/miem-refugees/sneakers-ml/actions/workflows/build-bot.yml/badge.svg)](https://github.com/miem-refugees/sneakers-ml/actions/workflows/build-bot.yml)
[![codecov](https://codecov.io/gh/miem-refugees/sneakers-ml/graph/badge.svg?token=ZTQC72KIPN)](https://codecov.io/gh/miem-refugees/sneakers-ml)

## [Цель проекта](https://docs.google.com/document/d/1Gdz3_W7x7L9Ff1-Sl61Cv3L6GHBiceH863Vn1ucXzjU/edit#heading=h.j88xs4dca7be)

Цель данного проекта - построить систему поиска похожих кроссовок по изображениям (задача-CV).
В проекте планируется реализовать парсинг данных, а именно картинок и дополнительных метаданных.
Далее следует этап чистки, обработки и объединения данных.
ML часть проекта будет заключатся в обучении классификаторов изображений кроссовок по брендам.
В DL части будет улучшено качество классификации с помощью продвинутных моделей, а так же решены другие задачи,
такие как image2image поиск и similarity learning.
В результате полученные модели будут обернуты в телеграм бот или streamlit сервис.

## Документация проекта

- [Презентации чекпоинтов](/docs/presentations)
- [Структура проекта](/docs/project-setup.md)
- [Описание данных](/docs/data-description.md)
- [Объединение данных и eda](/docs/eda-merging.md)
- [Описание моделей и эмбеддингов](/docs/features-models.md)
- [Инструкции по запуску](/docs/launch-instructions.md)

## Пример работы телеграм бота

![ezgif-3-a70e75c32f](https://github.com/miem-refugees/sneakers-ml/assets/57370975/0ded53d5-479d-458a-b1ed-3675b3e1f71c)

## Прогресс и задачи

- На текущий момент обучили лучшую модель классификации Resnet152 в
  [черновике-ноутбуке](notebooks/models/resnet_fine_tune.ipynb)
- Планируем раскидать код ноутбука по файлам, возможно использовать
  PyTorch Lightning
- Хотим попробовать Vision Transformer
- Далее будем решать другие задачи, такие как image2image поиск, similarity
  learning, text2image
- Улучшение telegram-бота, интеграция API с Streamlit
- Сборка датасета из изображений от пользователей через бота
- Логирование, мониторинг и алерты production-окружений (streamlit, tg-bot)

## Roadmap

- [x] **Поиск и сбор данных**
  - [x] Парсинг [sneakerbaas](https://www.sneakerbaas.com)
  - [x] Парсинг [superkicks](https://www.superkicks.in)
  - [x] Парсинг [highsnobiety](https://www.highsnobiety.com)
  - [x] Парсинг [kickscrew](https://www.kickscrew.com/)
  - [x] Парсинг [footshop](https://www.footshop.com)
  - [x] Выгрузка данных на s3 с помощью DVC
  - [x] Очистка данных
  - [x] Объединение данных в один датасет, готовый для тренировки моделей
  - [x] Документация и описание данных
- [x] **Настройка проекта**
  - [x] Настроить poetry
  - [x] Добавление линтеров и форматеров
  - [x] Добавление pre-commit
- [x] **Получение эмбеддингов**
  - [x] SIFT
  - [x] HOG
  - [x] ResNet152
  - [x] Сохранение в npy формате
- [x] **Классификация по брендам**
  - [x] Модели классического машинного обучения на полученных эмбеддингах
    - [x] SVM
    - [x] SGD
    - [x] CatBoost
    - [x] Сохранение в onnx формате
  - [ ] Модели глубинного обучения
    - [x] ResNet
    - [ ] Vision transformer
- [ ] **Обёртка моделей**
  - [x] FastAPI
  - [x] Telegram bot
  - [ ] streamlit
- [ ] image2image
  - [ ] faiss
- [ ] similarity learning
- [ ] Возможно text2image, image2text

## Установка

Для работы с API-сервисом или ботом установите необходимое окружение. Установите менеджер пакетов poetry:
```shell
curl -sSL https://install.python-poetry.org | python3 -
```

DVC: https://dvc.org/doc/install
Aws-cli S3 Яндекс облака: https://yandex.cloud/ru/docs/storage/tools/aws-cli

Запуск телеграм

## Список членов команды

- Литвинов Вячеслав
- Моисеев Даниил

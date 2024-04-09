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

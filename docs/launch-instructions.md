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

## Разработка сервисов

### Docker

В репозитории настроен per-commit flow в Github Actions, в котором собираются образы Api и Bot контейнеров.
На настоящее время результат выгружается в Yandex Container Registry (до этого в Github Registry)

| Api                                                   | Bot                                                   |
| ----------------------------------------------------- | ----------------------------------------------------- |
| cr.yandex/crp9sd2f3p1o3mfu9664/sneakers-ml-api:latest | cr.yandex/crp9sd2f3p1o3mfu9664/sneakers-ml-bot:latest |

Для простого поднятия сервисов создана docker-compose конфигурация:

- Local Docker-compose для локальной сборки: [docker-compose.local.yml](/docker-compose.yml)
- Docker-compose для production-развертывания: [docker-compose.yml](/deploy/docker-compose.yml)

Установите необходимое окружение:

- Poetry: <https://python-poetry.org/docs/#installation>
- Pre-commit: <https://pre-commit.com/#install>
- DVC: <https://dvc.org/doc/install>
- AWS CLI: <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>
- Docker [Compose]: <https://docs.docker.com/get-docker/>

> **_Важно:_** Перед началом работы получите нужные секреты скопировав `.env.example` в `.env` и заполнив его.
>
> ```shell
> $ cp .env.example .env
> ```

### **Docker Compose**

Одна команда поднимает весь сервис, включая API и бота:

```bash
docker-compose up
```

### **Ручной запуск**

Поднять бота:

```bash
make bot
```

Поднять API сервис:

```bash
make api
```

После этого сервис будет доступен по адресу `http://localhost:8000`.
Это является дефолтным значением, его можно изменить в `.env` файле.

Сервис API имеет страницу [Swagger UI](https://swagger.io). После запуска сервиса, она будет доступна по
адресу `http://localhost:8000/docs`.

## Тестирование

Для запуска тестов используйте команду:

```bash
make test
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

# Описание данных

## [superkicks](https://www.superkicks.in/)

### 8 категорий

- men-basketball-sneakers
- men-classics-sneakers
- men-skateboard-sneakers
- men-sneakers
- women-basketball-sneakers
- women-classics-sneakers
- women-skateboard-sneakers
- women-sneakers

### Фотографии

В основном каждая модель имеет несколько фотографий с различных углов:

| Вид            | Картинка                                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| Справа         | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/f2c43479-4069-42b7-87fe-12ead6d2943b) |
| Спереди        | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/1b8bb638-b1d2-48bb-b3de-e9ebcc16c907) |
| Сзади          | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/ebd62ffd-6ca5-4c39-b713-d6b4482f0dbf) |
| Передняя часть | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/dbc13ed8-54e3-4ece-9d2d-e49410a29500) |
| Задняя часть   | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/a025d0eb-f9cd-4bab-947f-723af3092ff5) |

Есть, модели в которых не хватает фотографий, есть те, где есть дополнительные. Сами ракурсы находятся в случайном порядке.

### Metadata

В папке [data/raw/metadata](/data/raw/metadata) содержится файл `superkicks.csv` с полями:

- **brand** - бренд
- **title** - название модели
- **price** - цена кроссовок
- **description** - описание
- **slug** - очищенное название модели
- **brand_slug** - очищенное название бренда
- **manufacturer** - производитель
- **country_of_origin** - страна-производитель
- **imported_by** - компания импортёр
- **weight** - вес кроссовок
- **generic_name**
- **unit_of_measurement** - количество пар
- **marketed_by** - название продавца
- **article_code** - код модели
- **collection_name** - название коллекции
- **collection_url** - ссылка на коллекцию
- **url** - ссылка на модель
- **images_path** - путь к локальной директории с картинками модели
- **product_dimensions**

## [sneakerbaas](https://www.sneakerbaas.com)

### 4 категории

- kids
- men
- unisex
- women

### Фотографии

В основном каждая модель кроссовок имеет по 3 фотографии на белом фоне.

| Вид     | Картинка                                                                                                    |
| ------- | ----------------------------------------------------------------------------------------------------------- |
| Слева   | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/7e728431-1238-4589-9563-9b9dd4d36960) |
| Справа  | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/a590de7f-e0f2-47e0-825c-2b4ff788a2e4) |
| Подошва | ![image](https://github.com/miem-refugees/sneakers-ml/assets/57370975/22d2e9e8-df51-4c3b-89b9-c21b0f15ae2e) |

Есть, модели в которых не хватает фотографий, есть те, где есть дополнительные. Сами ракурсы находятся в случайном порядке.

### Metadata

В папке [data/raw/metadata](/data/raw/metadata) содержится файл `sneakerbaas.csv` с полями:

- **brand** - бренд
- **description** - описание, из которого можно достадь дополнительную информацию (цвет модели и т.д.)
- **pricecurrency** - валюта, в которой продаются кроссовки
- **price** - цена кроссовок
- **title** - название модели
- **slug** - очищенное название модели
- **brand_slug** - очищенное название бренда
- **collection_name** - название коллекции
- **collection_url** - ссылка на коллекцию
- **url** - ссылка на модель
- **images_path** - путь к локальной директории с картинками модели

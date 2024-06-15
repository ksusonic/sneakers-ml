# Data splits

- Работаем с классификацией по брендам
- У некоторых брендов мало картинок.
- Отобрали только те бренды, у которых больше 100 картинок.
- Получилось 13 брендов для классификации.
- Разделили данные на train/val/test в пропорциях 60/20/20.

## Features and models

### Base dataset

| Feature-model_name      | f1_macro | f1_micro | f1_weighted | accuracy |
| ----------------------- | -------- | -------- | ----------- | -------- |
| Baseline                | 0.03     | 0.29     | 0.13        | 0.29     |
| hog-log_reg             | 0.72     | 0.76     | 0.76        | 0.76     |
| hog-random_forest       | 0.54     | 0.66     | 0.63        | 0.66     |
| hog-decision_tree       | 0.34     | 0.45     | 0.45        | 0.45     |
| hog-svm                 | 0.78     | 0.81     | 0.81        | 0.81     |
| hog-sgd                 | 0.72     | 0.76     | 0.75        | 0.76     |
| hog-catboost            | 0.67     | 0.73     | 0.71        | 0.73     |
| sift-log_reg            | 0.29     | 0.38     | 0.36        | 0.38     |
| sift-random_forest      | 0.18     | 0.38     | 0.29        | 0.38     |
| sift-decision_tree      | 0.17     | 0.24     | 0.24        | 0.24     |
| sift-svm                | 0.37     | 0.47     | 0.44        | 0.47     |
| sift-sgd                | 0.26     | 0.34     | 0.34        | 0.34     |
| sift-catboost           | 0.29     | 0.44     | 0.39        | 0.44     |
| resnet152-log_reg       | 0.71     | 0.73     | 0.73        | 0.73     |
| resnet152-random_forest | 0.41     | 0.56     | 0.51        | 0.56     |
| resnet152-decision_tree | 0.27     | 0.38     | 0.36        | 0.38     |
| resnet152-svm           | 0.76     | 0.76     | 0.76        | 0.76     |
| resnet152-sgd           | 0.72     | 0.74     | 0.74        | 0.74     |
| resnet152-catboost      | 0.64     | 0.7      | 0.68        | 0.7      |

### Removed подошвы с помощью UMAP

| Feature-model_name      | f1_macro | f1_micro | f1_weighted | accuracy |
| ----------------------- | -------- | -------- | ----------- | -------- |
| Baseline                | 0.03     | 0.29     | 0.13        | 0.29     |
| hog-log_reg             | 0.73     | 0.77     | 0.77        | 0.77     |
| hog-random_forest       | 0.51     | 0.66     | 0.62        | 0.66     |
| hog-decision_tree       | 0.34     | 0.44     | 0.44        | 0.44     |
| hog-svm                 | 0.79     | 0.82     | 0.82        | 0.82     |
| hog-sgd                 | 0.71     | 0.76     | 0.75        | 0.76     |
| hog-catboost            | 0.66     | 0.73     | 0.71        | 0.73     |
| sift-log_reg            | 0.28     | 0.39     | 0.37        | 0.39     |
| sift-random_forest      | 0.15     | 0.41     | 0.3         | 0.41     |
| sift-decision_tree      | 0.16     | 0.26     | 0.25        | 0.26     |
| sift-svm                | 0.36     | 0.46     | 0.43        | 0.46     |
| sift-sgd                | 0.25     | 0.38     | 0.36        | 0.38     |
| sift-catboost           | 0.32     | 0.48     | 0.43        | 0.48     |
| resnet152-log_reg       | 0.74     | 0.75     | 0.75        | 0.75     |
| resnet152-random_forest | 0.4      | 0.57     | 0.51        | 0.57     |
| resnet152-decision_tree | 0.28     | 0.36     | 0.36        | 0.36     |
| resnet152-svm           | 0.74     | 0.75     | 0.75        | 0.75     |
| resnet152-sgd           | 0.67     | 0.69     | 0.7         | 0.69     |
| resnet152-catboost      | 0.64     | 0.71     | 0.69        | 0.71     |

### DL на расширенном датасете (с footshop)

todo

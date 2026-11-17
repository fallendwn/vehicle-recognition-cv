# Классификация одежды FashionMNIST

Учебный проект по компьютерному зрению. Свёрточная нейронная сеть классифицирует
чёрно-белые изображения FashionMNIST по 10 классам:

`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`,
`Sneaker`, `Bag`, `Ankle boot`.

В проекте **нет data augmentation**. Для train, validation и test используется
одинаковая предобработка:

```python
transforms.ToTensor()
transforms.Normalize((0.2860,), (0.3530,))
```

Это стандартные приближённые mean и standard deviation обучающей выборки
FashionMNIST. Из исходного training split воспроизводимо выбираются ровно 10 000
изображений — по 1 000 каждого класса. Они стратифицированно делятся на 8 000
train (по 800 на класс) и 2 000 validation (по 200 на класс). Официальные 10 000
test-изображений не участвуют в выборе модели и также содержат по 1 000 примеров
каждого класса.

## Структура

```text
train.py              загрузка FashionMNIST, обучение и оценка CNN
model.py              архитектура FashionMNISTCNN
fine_tune.py          дополнительное обучение сохранённой модели
data/                 FashionMNIST (скачивается автоматически)
models/               лучший checkpoint
reports/              метрики и confusion matrix
```

## Установка

Рекомендуется Python 3.11 или 3.12:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Обучение

```powershell
python train.py --epochs 15 --batch-size 128
```

При первом запуске `torchvision` автоматически скачает FashionMNIST в `data/`.
На NVIDIA CUDA будет использована видеокарта, иначе обучение пройдёт на CPU.

Результаты:

- `models/fashion_mnist_cnn.pt` — модель с лучшей validation accuracy;
- `reports/fashion_mnist_metrics.json` — история обучения и test-метрики;
- `reports/fashion_mnist_confusion_matrix.png` — матрица ошибок.

Дополнительные параметры:

```powershell
python train.py --help
```

Размер сбалансированной подвыборки можно изменить параметрами
`--samples-per-class` и `--validation-per-class`.

## Датасет

FashionMNIST доступен напрямую через `torchvision.datasets.FashionMNIST` и
содержит 60 000 обучающих и 10 000 тестовых изображений размером 28×28.

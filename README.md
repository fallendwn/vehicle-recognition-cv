# Классификация транспорта на фотографии

Учебный проект по компьютерному зрению. Пользователь загружает изображение в браузере, а модель MobileNetV3 определяет один из пяти типов транспорта:

- `bicycle` — велосипед;
- `bus` — автобус;
- `car` — автомобиль;
- `motorcycle` — мотоцикл;
- `truck` — грузовик.

Модель обучается методом transfer learning: используется предобученная MobileNetV3 Small, а последний классификационный слой обучается на собственном наборе изображений из Open Images V7.

## Структура проекта

```text
app/                      FastAPI и веб-интерфейс
download_open_images.py   загрузка и подготовка датасета
prepare_augmented_dataset.py  сохранение изменённых копий и балансировка train
train.py                  обучение и оценка модели
dataset/                  train/val/test (создаётся автоматически)
dataset_augmented/        отдельные изменённые копии только для train
models/                   обученная модель
reports/                  метрики и confusion matrix
```

## 1. Установка

Рекомендуется Python 3.11 или 3.12. В PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Подготовка датасета

FiftyOne нужен только для скачивания Open Images:

```powershell
pip install -r requirements-data.txt
python download_open_images.py --images-per-class 200
```

Скрипт скачивает изображения с разметкой, выбирает крупнейший объект нужного класса, вырезает его по bounding box и создаёт разбиение 70/15/15:

```text
dataset/
├── train/{car,bus,truck,motorcycle,bicycle}
├── val/{car,bus,truck,motorcycle,bicycle}
└── test/{car,bus,truck,motorcycle,bicycle}
```

После скачивания рекомендуется вручную просмотреть изображения и удалить неверно размеченные примеры. Не переносите один и тот же файл между train и test.

Перед обучением можно выровнять число обучающих примеров. Скрипт не меняет исходные фотографии и не добавляет изменённые копии в `val` или `test`. Он создаёт отражённые, повёрнутые, обрезанные и скорректированные по цвету копии в отдельной папке:

```powershell
python prepare_augmented_dataset.py
```

По умолчанию каждый класс дополняется до размера самого крупного класса. Параметры запуска и происхождение каждой копии записываются в `dataset_augmented/manifest.json`.

## 3. Обучение

```powershell
python train.py --epochs 15 --batch-size 32
```

Если существует `dataset_augmented/train`, сохранённые копии автоматически добавляются только к обучающей выборке. Их можно отключить параметром `--augmented-data` с путём к несуществующему каталогу.

На компьютере с NVIDIA CUDA будет использована видеокарта, иначе обучение пройдёт на CPU. Результаты:

- `models/transport_classifier.pt` — лучшая модель по validation accuracy;
- `reports/metrics.json` — accuracy, precision, recall и F1;
- `reports/confusion_matrix.png` — матрица ошибок.

## 4. Запуск сайта

При желании выполните дополнительный fine-tuning последних блоков MobileNetV3:

```powershell
python fine_tune.py --epochs 10 --batch-size 16
```

Результат сохраняется в `models/transport_classifier_finetuned.pt`. Сайт автоматически использует fine-tuned checkpoint, если он существует.

```powershell
uvicorn app.main:app --reload
```

Откройте <http://127.0.0.1:8000>. Документация API доступна на <http://127.0.0.1:8000/docs>.

Проверка состояния модели:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

## API

`POST /api/predict` принимает поле `file` формата JPG, PNG или WebP размером до 10 МБ:

```json
{
  "label": "bus",
  "confidence": 0.9321,
  "probabilities": [
    {"label": "bus", "confidence": 0.9321},
    {"label": "truck", "confidence": 0.0412}
  ]
}
```

## Источник данных

Open Images V7: <https://storage.googleapis.com/openimages/web/index.html>. Аннотации распространяются по CC BY 4.0; изображения указаны как CC BY 2.0. Для учебного отчёта укажите источник датасета и MobileNetV3 из `torchvision`.


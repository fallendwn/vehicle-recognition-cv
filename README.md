# Классификация транспорта с DINOv2

Учебный проект по компьютерному зрению для Assignment 2. Пользователь загружает
изображение или делает снимок в браузере, а DINOv2 определяет один из пяти типов
транспорта:

- `bicycle` — велосипед;
- `bus` — автобус;
- `car` — автомобиль;
- `motorcycle` — мотоцикл;
- `truck` — грузовик.

Assignment 1 с MobileNetV3 Small сохранён как baseline. В Assignment 2 используется
официальный DINOv2 ViT-S/14 от Meta: backbone заморожен, а на собственном наборе
изображений из Open Images V7 обучается линейный классификатор. Приложение также
показывает patch-relevance heatmap, ближайшие изображения в пространстве DINOv2
embeddings и время инференса.

## Структура проекта

```text
app/                      FastAPI и веб-интерфейс
download_open_images.py   загрузка и подготовка датасета
prepare_augmented_dataset.py  сохранение изменённых копий и балансировка train
train.py                  обучение и оценка модели
train_dinov2.py           DINOv2, ablation и сравнение с baseline
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

- `models/transport_classifier.pt` — baseline Assignment 1;
- `reports/metrics.json` — accuracy, precision, recall и F1;
- `reports/confusion_matrix.png` — матрица ошибок.

## 4. Обучение DINOv2 и ablation

Первый запуск скачивает официальный DINOv2 ViT-S/14 через PyTorch Hub. Backbone
остаётся замороженным. Скрипт извлекает embeddings один раз, обучает два линейных
классификатора и сравнивает их на одном test split:

```powershell
python train_dinov2.py --epochs 150 --batch-size 16
```

Эксперименты:

- `no_augmentation` — только исходный train split;
- `with_augmentation` — исходный train split и сохранённые аугментированные копии.

Результаты сохраняются в `reports/dinov2_metrics.json`. Скрипт также создаёт
confusion matrix, validation curves, график сравнения с MobileNetV3, checkpoint
линейной головы и retrieval index. Полные веса DINOv2 не копируются в репозиторий:
их загружает официальный PyTorch Hub.

## 5. Запуск сайта

При желании выполните дополнительный fine-tuning последних блоков MobileNetV3:

```powershell
python fine_tune.py --epochs 10 --batch-size 16
```

Результат сохраняется в `models/transport_classifier_finetuned.pt` и используется
только как baseline для сравнения. Веб-приложение Assignment 2 использует
`models/dinov2_transport_classifier.pt`.

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
  "latency_ms": 142.7,
  "device": "CPU",
  "heatmap": "data:image/jpeg;base64,...",
  "neighbors": [
    {"index": 31, "label": "bus", "similarity": 0.91, "url": "/api/reference/31"}
  ],
  "probabilities": [
    {"label": "bus", "confidence": 0.9321},
    {"label": "truck", "confidence": 0.0412}
  ]
}
```

## Источник данных

Open Images V7: <https://storage.googleapis.com/openimages/web/index.html>.
Аннотации распространяются по CC BY 4.0; изображения указаны как CC BY 2.0.

DINOv2:

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023/2024: <https://arxiv.org/abs/2304.07193>;
- официальная реализация и model card: <https://github.com/facebookresearch/dinov2>.

## Воспроизводимость

- random seed: `42`;
- одинаковые validation и test splits для MobileNetV3 и DINOv2;
- test split не используется при выборе checkpoint;
- synthetic media не используется;
- сохранённые аугментации добавляются только в train split;
- параметры, история обучения, class-wise metrics и latency сохраняются в JSON.


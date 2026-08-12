# CIFAR-10 Object Classifier

> CNN распознаёт реальные объекты по 10 категориям — автоматизация
> визуальной инспекции и классификации изображений в production.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal)]()
[![Accuracy](https://img.shields.io/badge/Train-79.11%25_|_Test-78.09%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

Ручная классификация изображений в логистике, ритейле и модерации
контента — медленная и дорогостоящая. Этот API классифицирует
изображение в момент загрузки без участия человека.

---

## Структура проекта

    ml_CIFAR_10/
    ├── .gitignore
    ├── readme.md
    ├── requirements.txt
    └── cifar_10/
        ├── CIFAR_10.ipynb                        ← обучение (Google Colab, GPU T4)
        ├── main.py                               ← FastAPI сервер + inference
        ├── model_CifarClassification_CIFAR_10.pth
        └── cifar_10 test/                        ← тестовые изображения
            ├── airplane.png
            ├── automobile.png
            ├── bird.png
            ├── cat.png
            ├── deer.png
            ├── dog.png
            ├── frog.png
            ├── horse.png
            ├── ship.png
            └── truck.png

---

## Быстрый старт

```bash
git clone https://github.com/your-username/ml_CIFAR_10
cd ml_CIFAR_10/cifar_10
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs`

---

## Demo

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@dog.png"
```

```json
{
  "class": "dog"
}
```

---

## Результаты

| Модель                        | Train Accuracy | Test Accuracy |
|-------------------------------|----------------|---------------|
| Random classifier (10 кл.)    | 10%            | 10%           |
| Linear (flatten only)         | ~55%           | ~55%          |
| **Custom CNN (3-block)**      | **79.11%**     | **78.09%**    |

Обучение: 25 эпох, Adam lr=0.001, ReduceLROnPlateau,
batch_size=64, аугментация (RandomFlip + RandomCrop), GPU T4.

**Почему кастомный CNN, а не ResNet / EfficientNet:**
- CIFAR-10 — 32×32 RGB, не фото реального мира высокого разрешения
- 3-block CNN весит < 5 МБ, инференс < 30 мс на CPU
- 78% на сбалансированном датасете достаточно для базового пайплайна
- Разрыв train/test (79%/78%) — минимальный: аугментация сработала

---

## Датасет

- **Источник:** CIFAR-10 (Alex Krizhevsky) — загружается через `torchvision`
- **Объём:** 60 000 RGB-изображений (50K train / 10K test)
- **Размер:** 32×32 px, 3 канала
- **Баланс:** 6 000 примеров на класс — ресэмплинг не нужен

| ID | Класс       | ID | Класс       |
|----|-------------|----|-------------|
| 0  | airplane    | 5  | dog         |
| 1  | automobile  | 6  | frog        |
| 2  | bird        | 7  | horse       |
| 3  | cat         | 8  | ship        |
| 4  | deer        | 9  | truck       |

---

## Архитектура модели

    Conv2d(3→32) + ReLU + MaxPool2d(2)
              ↓
    Conv2d(32→64) + ReLU + MaxPool2d(2)
              ↓
    Conv2d(64→128) + ReLU + MaxPool2d(2)
              ↓
    Flatten → Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→10)
              ↓
    argmax → class name

**Ключевые решения:**

`RandomHorizontalFlip + RandomCrop(32, padding=4)` в train-трансформе —
аугментация снижает переобучение без дополнительных слоёв регуляризации.
Разрыв train/test составил всего 1%, что подтверждает эффективность.

`Dropout(0.5)` в классификаторе — предотвращает запоминание обучающей
выборки на глубоких признаках.

`ReduceLROnPlateau(factor=0.5, patience=3)` — автоматически снижает
learning rate при стагнации loss, позволяя дообучиться без ручного тюнинга.

`Resize((32, 32)) + ToTensor()` в inference — реальные изображения
приходят в разных размерах и форматах. Ресайз на стороне сервера
устраняет ошибки несовпадения размерности без требований к клиенту.

---

## Стек

| Слой    | Технологии                              |
|---------|-----------------------------------------|
| ML      | PyTorch, torchvision, Pillow            |
| API     | FastAPI, Uvicorn                        |
| Обучение | Google Colab (GPU T4), Jupyter         |

---

## Business Impact

| Задача                           | До                         | После                    |
|----------------------------------|----------------------------|--------------------------|
| Классификация одного объекта     | 2–5 мин ручной работы      | < 30 мс на запрос        |
| Последовательность тегов         | Зависит от оператора       | Детерминированная модель |
| Масштабирование                  | Линейно к числу операторов | Один REST-вызов          |

---

## Что дальше (Roadmap)

- [ ] `confidence` в ответе — вернуть softmax вероятность вместе с классом
- [ ] Docker + Nginx — production-деплой по аналогии с MNIST Fashion
- [ ] Более глубокая аугментация — color jitter, cutout → цель +5–7% accuracy
- [ ] MLflow — трекинг экспериментов и версионирование модели
- [ ] Fine-tuning ResNet18 на CIFAR-10 — потолок кастомного CNN ~80–82%

---

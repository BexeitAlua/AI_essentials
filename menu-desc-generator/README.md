# menu-desc-generator — генератор описаний блюд

[![tests](https://github.com/<ваш_ник>/menu-desc-generator/actions/workflows/tests.yml/badge.svg)](https://github.com/<ваш_ник>/menu-desc-generator/actions/workflows/tests.yml)

**Живой сервис:** https://<ваш_ник>-menu-desc-generator.hf.space ·
**Модель:** https://huggingface.co/<ваш_ник>/rut5-menu-desc (тег `v1.0.0`) ·
**Автор:** Бексеит Алуа, AAI-2601


Прототип интеллектуальной системы: обученная модель, API, веб-интерфейс, тесты, мониторинг, деплой.
По названию, категории и ингредиентам блюда модель `rut5-menu-desc@1.0.0` (ruT5-base, дообученная
в Midterm) пишет короткое описание для карточки меню. Подробно о модели — в `MODEL_CARD.md`.

## Архитектура

```
Браузер (веб-форма)  ──┐
                       ├─► FastAPI (app/main.py) ──► MenuDescriptionGenerator (menu_desc/)
Клиент API (JSON)    ──┘        │                       валидация → ruT5-base INT8 → проверки текста
                                ├─ /health, /version
                                └─ /metrics  ◄── Metrics + JSON-логи каждого запроса (app/monitoring.py)

Hugging Face Hub:  <user>/rut5-menu-desc (веса + model_version.json, тег v1.0.0)
                   <user>/menu-desc-generator (Space, Docker) — скачивает модель по тегу при старте
```

## Структура
```
menu_desc/preprocessing.py   валидация и нормализация входа, сборка строки в формате обучения, warnings
menu_desc/schemas.py         контракт входа/выхода (pydantic)
menu_desc/model.py           MenuDescriptionGenerator: проверка SHA-256, загрузка, eval, генерация, INT8
app/main.py                  FastAPI: /predict, /predict/batch, /health, /version, /metrics, веб-форма
app/monitoring.py            счётчики, задержки p50/p95, доля ошибок и предупреждений, JSON-логи
app/static/index.html        веб-интерфейс (форма + панель мониторинга)
scripts/prepare_final_model.py  фиксация версии модели: models/rut5-menu-desc/<version>/
notebooks/Midterm_extended.ipynb  эксперименты и обучение моделей (Midterm)
notebooks/Task1_Final_Model.ipynb  Задание 1: подготовка финальной модели (Colab)
notebooks/Deploy_HF_Spaces.ipynb   деплой модели и сервиса на Hugging Face Spaces (Colab)
tests/                       pytest: 47 тестов (модуль модели, предобработка, API)
Dockerfile                   образ сервиса (CPU, порт 7860)
```

## API

| Метод | Путь | Описание |
|---|---|---|
| POST | `/predict` | одно блюдо → описание |
| POST | `/predict/batch` | `{"dishes": [...]}`, до 16 блюд |
| GET | `/health` | `model_state`: `loading` / `ready` / `error` |
| GET | `/version` | версия модели, хэш весов, параметры генерации, метрики на test |
| GET | `/metrics` | мониторинг (JSON или `?format=prometheus`) |
| GET | `/docs` | Swagger UI |

Пример:
```bash
curl -X POST https://<user>-menu-desc-generator.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Бешбармак", "category": "Горячие блюда > Блюда из мяса",
       "ingredients": "конина, баранина, тесто, лук"}'
```
```json
{"description": "…", "model_version": "rut5-menu-desc@1.0.0", "warnings": [],
 "input_tokens": 33, "latency_ms": 860.0}
```
Вход: `title` (1–100 символов, обязательно), `category` (строка «A > B» или список, необязательно),
`ingredients` (строка через запятую или список, 1–30). Некорректный вход → HTTP 422 с текстом ошибки.

`warnings` — продуктовые проверки ответа: `health_claim` («полезный», «лучший»…), `first_person`,
`no_ingredient_mentioned`, `empty_output`, `input_truncated`.

## Мониторинг
- `/metrics`: число запросов, доля ошибок (ввод / сервер), задержка p50 / p95, доля описаний с каждым warning;
- каждый запрос и каждая генерация пишутся в лог JSON-строкой (вкладка **Logs** у Space);
- `/health` показывает состояние модели; веб-интерфейс отображает метрики в реальном времени.

## Деплой (Hugging Face Spaces, бесплатно)
1. Выполнить `Task1_Final_Model.ipynb` → версия модели на Drive (`task1_models/rut5-menu-desc/1.0.0`).
2. Создать токен Hugging Face с правом Write.
3. Выполнить `Deploy_HF_Spaces.ipynb`: загрузит модель с тегом `v1.0.0`, создаст Space, дождётся
   сборки и проверит сервис.

Переменные окружения сервиса: `MODEL_REPO`, `MODEL_REVISION` (тег версии), `MODEL_DIR`,
`QUANTIZE_INT8` (по умолчанию `true`), `NUM_THREADS`, `MAX_BATCH`, `HF_TOKEN` (для приватной модели).

## Локальный запуск
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt
MODEL_DIR=models/rut5-menu-desc/1.0.0 uvicorn app.main:app --port 7860   # http://localhost:7860
```

## Тесты
```bash
pip install -r requirements.txt fastapi httpx
pytest -q          # 47 тестов, крошечная T5 вместо 900-МБ весов, ~15 с
```

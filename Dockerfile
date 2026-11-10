# Сервис генерации описаний блюд: FastAPI + ruT5-base (INT8 на CPU).
# Hugging Face Spaces (sdk: docker) собирает этот образ и открывает порт 7860.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

# HF Spaces запускает контейнер от пользователя с UID 1000
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
WORKDIR /home/user/app

# CPU-сборка PyTorch: ~200 МБ вместо ~2.5 ГБ CUDA-сборки
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY --chown=user menu_desc ./menu_desc
COPY --chown=user app ./app
USER user

# Модель скачивается при старте из репозитория MODEL_REPO (тег MODEL_REVISION) — задаются в Settings у Space.
# SHA-256 весов сверяется с model_version.json, затем модель квантуется в INT8.
ENV MODEL_DIR=/home/user/app/models/rut5-menu-desc \
    MODEL_REVISION=v1.0.0 \
    QUANTIZE_INT8=true \
    NUM_THREADS=2

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

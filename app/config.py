"""Настройки сервиса из переменных окружения (их задают в Dockerfile или в Settings у Space)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_dir: Path                 # где лежит версия модели (model_version.json + веса)
    model_repo: str | None          # репозиторий модели на HF Hub, например "username/rut5-menu-desc"
    model_revision: str             # ветка / тег / коммит репозитория — какая версия попадёт в сервис
    hf_token: str | None            # нужен только для приватного репозитория модели
    quantize_int8: bool             # INT8 на CPU: в 3 раза быстрее FP32 при том же качестве
    num_threads: int
    max_batch: int


def load_settings() -> Settings:
    return Settings(
        model_dir=Path(os.getenv("MODEL_DIR", "models/rut5-menu-desc/1.0.0")),
        model_repo=os.getenv("MODEL_REPO") or None,
        model_revision=os.getenv("MODEL_REVISION", "main"),
        hf_token=os.getenv("HF_TOKEN") or None,
        quantize_int8=_bool(os.getenv("QUANTIZE_INT8", "true")),
        num_threads=int(os.getenv("NUM_THREADS", "2")),
        max_batch=int(os.getenv("MAX_BATCH", "16")),
    )

"""Inference финальной модели: ruT5-base (fine-tuned) + подобранные параметры декодирования.

Использование:
    gen = MenuDescriptionGenerator.from_dir("models/rut5-menu-desc/1.0.0")
    out = gen.predict({"title": "Бешбармак", "category": ["Горячие блюда", "Блюда из мяса"],
                       "ingredients": ["конина", "баранина", "тесто", "лук репчатый", "бульон"]})
    print(out.description, out.model_version, out.warnings)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .preprocessing import (InputValidationError, build_source, normalize_category, normalize_ingredients,
                            normalize_title, output_warnings)
from .schemas import DescriptionOutput, DishInput

logger = logging.getLogger("menu_desc")

VERSION_FILE = "model_version.json"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _select_quantized_engine() -> str | None:
    """Движок INT8: x86/fbgemm на серверах Linux, qnnpack на ARM (Apple silicon). None — нет ни одного."""
    current = torch.backends.quantized.engine
    if current and current != "none":
        return current
    for engine in ("x86", "fbgemm", "qnnpack", "onednn"):
        if engine in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = engine
            return engine
    return None


class ModelIntegrityError(RuntimeError):
    """Веса не совпадают с зафиксированной версией."""


class MenuDescriptionGenerator:
    def __init__(self, model, tokenizer, version_info: dict[str, Any], device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.info = version_info
        self.device = device
        self.version = f"{version_info['name']}@{version_info['version']}"
        self.max_source_len = int(version_info["input"]["max_source_tokens"])
        self.generation_kwargs = dict(version_info["generation"])

    # ------------------------------------------------------------------ загрузка
    @classmethod
    def from_dir(cls, model_dir: Union[str, Path], device: str | None = None, verify_hash: bool = True,
                 quantize_int8: bool = False, num_threads: int | None = None) -> "MenuDescriptionGenerator":
        model_dir = Path(model_dir)
        info_path = model_dir / VERSION_FILE
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} не найден: подготовьте модель скриптом prepare_final_model.py")
        info = json.loads(info_path.read_text(encoding="utf-8"))

        if verify_hash:
            for fname, expected in info["files_sha256"].items():
                if fname in WEIGHT_FILES:
                    actual = sha256_file(model_dir / fname)
                    if actual != expected:
                        raise ModelIntegrityError(
                            f"SHA-256 {fname} = {actual[:12]}…, ожидалось {expected[:12]}… — это не версия {info['version']}")

        if num_threads:
            torch.set_num_threads(num_threads)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.float32)
        model.eval()
        quantization = "none"
        if quantize_int8:
            if device != "cpu":
                raise ValueError("Динамическое INT8-квантование поддерживается только на CPU")
            engine = _select_quantized_engine()
            if engine is None:
                logger.warning("INT8: нет движка квантования в этой сборке PyTorch — работаем в FP32")
            else:
                try:
                    # квантуются только Linear-слои; веса хранятся в INT8, вычисления на CPU
                    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    quantization = f"dynamic-int8 ({engine})"
                except RuntimeError as e:     # например, macOS-сборка без поддержки нужного оператора
                    logger.warning("INT8 не удалось (%s) — работаем в FP32", e)
        info = {**info, "runtime": {**info.get("runtime", {}), "quantization": quantization}}
        model.to(device)
        logger.info("Loaded %s@%s on %s", info["name"], info["version"], device)
        return cls(model, tokenizer, info, device)

    # ------------------------------------------------------------------ вход
    def prepare(self, dish: Union[DishInput, dict]) -> tuple[str, list[str]]:
        """Проверяет и нормализует вход; возвращает строку для модели и список ингредиентов."""
        if isinstance(dish, dict):
            dish = DishInput(**dish)
        title = normalize_title(dish.title)
        category = normalize_category(dish.category)
        ingredients = normalize_ingredients(dish.ingredients)
        return build_source(title, category, ingredients), ingredients

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def generate_raw(self, sources: list[str]) -> tuple[list[str], float]:
        """Генерация по готовым строкам входа (формат build_source). Нужна для сверки с test-предсказаниями
        ноутбука; в продукте используйте predict / predict_batch — они проверяют вход."""
        enc = self.tokenizer(sources, max_length=self.max_source_len, truncation=True,
                             padding=True, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        generated = self.model.generate(**enc, **self.generation_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000 / len(sources)
        return [t.strip() for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)], latency_ms

    def predict_batch(self, dishes: Iterable[Union[DishInput, dict]]) -> list[DescriptionOutput]:
        prepared = [self.prepare(d) for d in dishes]
        if not prepared:
            return []
        sources = [p[0] for p in prepared]
        full_lens = [len(self.tokenizer(s)["input_ids"]) for s in sources]
        texts, latency_ms = self.generate_raw(sources)

        outputs = []
        for text, (_, ingredients), n_tok in zip(texts, prepared, full_lens):
            warnings = output_warnings(text, ingredients)
            if n_tok > self.max_source_len:
                warnings.append("input_truncated")
            outputs.append(DescriptionOutput(description=text, model_version=self.version, warnings=warnings,
                                             input_tokens=min(n_tok, self.max_source_len),
                                             latency_ms=round(latency_ms, 1)))
        return outputs

    def predict(self, dish: Union[DishInput, dict]) -> DescriptionOutput:
        return self.predict_batch([dish])[0]

    def describe(self) -> dict[str, Any]:
        """Метаданные для эндпоинта /version и логов мониторинга."""
        return {"model_version": self.version, "device": self.device,
                "base_model": self.info["base_model"], "generation": self.generation_kwargs,
                "weights_sha256": self.info["files_sha256"].get("model.safetensors"),
                "quantization": self.info.get("runtime", {}).get("quantization", "none"),
                "created_at": self.info["created_at"]}


__all__ = ["MenuDescriptionGenerator", "InputValidationError", "ModelIntegrityError", "sha256_file"]

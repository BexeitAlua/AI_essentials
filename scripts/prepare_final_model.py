"""Подготовка финальной модели к inference: копирование весов в версионированную папку,
фиксация параметров генерации, хэши файлов, метаданные эксперимента.

Запуск (в той же среде, где обучался ноутбук Midterm, например в Colab):
    python scripts/prepare_final_model.py --src results/rut5_menu_desc \
        --dst models/rut5-menu-desc/1.0.0 --results results/final_results.csv --env results/environment.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from menu_desc.model import VERSION_FILE, sha256_file  # noqa: E402

MODEL_NAME = "rut5-menu-desc"
SELECTED_EXPERIMENT = "ruT5-base + tuned decoding"

# Финальная конфигурация = веса baseline ruT5 (lr 5e-4, batch 8, 2 эпохи) + лучшие параметры декодирования,
# подобранные на validation (Midterm, раздел 9b). Эти значения — часть версии модели: меняются только вместе с ней.
GENERATION = {
    "num_beams": 4,
    "no_repeat_ngram_size": 2,
    "length_penalty": 2.0,
    "min_new_tokens": 15,
    "max_new_tokens": 80,
    "early_stopping": True,
    "do_sample": False,
}
TRAINING = {"base_model": "ai-forever/ruT5-base", "dataset": "d0rj/povarenok_recipes_detail",
            "train_size": 10_000, "epochs": 2, "batch_size": 8, "learning_rate": 5e-4,
            "optimizer": "adafactor", "warmup_ratio": 0.05, "seed": 42, "max_source_len": 128,
            "max_target_len": 80}
KEEP = ("config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin", "spiece.model",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


def read_metrics(results_csv: Path | None) -> dict:
    if not results_csv or not results_csv.exists():
        return {}
    with open(results_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model") == SELECTED_EXPERIMENT:
                keys = ["chrF++", "chrF++_b1", "ROUGE-L", "BERTScore_F1", "ingredient_mention_%",
                        "health_claim_%", "avg_chars"]
                return {k: float(row[k]) for k in keys if row.get(k)}
    return {}


def read_best_decoding(experiments_json: Path | None) -> dict:
    """Лучшие параметры декодирования из experiments.json (Midterm) — чтобы версия совпала с экспериментом."""
    if not experiments_json or not experiments_json.exists():
        return {}
    best = json.loads(experiments_json.read_text(encoding="utf-8")).get("best", {}).get("t5_decoding", {})
    casts = {"num_beams": int, "no_repeat_ngram_size": int, "length_penalty": float, "min_new_tokens": int}
    return {k: casts[k](v) for k, v in best.items() if k in casts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="папка, сохранённая trainer.save_model()")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--results", type=Path, help="final_results.csv из ноутбука (метрики на test)")
    ap.add_argument("--env", type=Path, help="environment.json из ноутбука (версии библиотек)")
    ap.add_argument("--experiments", type=Path, help="experiments.json из ноутбука (лучшее декодирование)")
    args = ap.parse_args()

    best = read_best_decoding(args.experiments)
    if best and any(GENERATION[k] != v for k, v in best.items()):
        print(f"Параметры декодирования взяты из {args.experiments}: {best}")
    GENERATION.update(best)

    if args.dst.exists() and any(args.dst.iterdir()):
        sys.exit(f"{args.dst} уже существует: версии не перезаписываются, увеличьте --version")
    args.dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in KEEP:
        if (args.src / name).exists():
            shutil.copy2(args.src / name, args.dst / name)
            copied.append(name)
    if not any(w in copied for w in ("model.safetensors", "pytorch_model.bin")):
        sys.exit(f"В {args.src} нет файла весов")

    # generation_config.json: даже обычный model.generate() без аргументов использует финальные параметры
    gen_cfg_path = args.dst / "generation_config.json"
    gen_cfg = json.loads(gen_cfg_path.read_text()) if gen_cfg_path.exists() else {}
    gen_cfg.update(GENERATION)
    gen_cfg_path.write_text(json.dumps(gen_cfg, indent=2))
    if "generation_config.json" not in copied:
        copied.append("generation_config.json")

    env = json.loads(args.env.read_text(encoding="utf-8")) if args.env and args.env.exists() else {}
    info = {
        "name": MODEL_NAME,
        "version": args.version,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "selected_experiment": SELECTED_EXPERIMENT,
        "base_model": TRAINING["base_model"],
        "task": "data-to-text: поля блюда → короткое описание для карточки меню (ru)",
        "training": TRAINING,
        "generation": GENERATION,
        "input": {
            "format": "Блюдо: {title}\\nКатегория: {cat1 > cat2}\\nИнгредиенты: {i1, i2, ...}",
            "fields": {"title": "str, 1–100 символов, обязательно",
                       "category": "str 'A > B' или list[str], опционально, берутся 2 уровня",
                       "ingredients": "list[str] или str через запятую, 1–30 шт., ≤ 60 символов каждый"},
            "max_source_tokens": TRAINING["max_source_len"],
            "language": "ru",
        },
        "output": {"description": "str, русский текст, обычно 1–2 предложения (≈ 70 символов на test)",
                   "warnings": ["health_claim", "first_person", "no_ingredient_mentioned", "empty_output",
                                "input_truncated"]},
        "test_metrics": read_metrics(args.results),
        "environment": {k: env[k] for k in ("python", "torch", "transformers", "sentencepiece") if k in env},
        "files_sha256": {name: sha256_file(args.dst / name) for name in sorted(copied)},
    }
    (args.dst / VERSION_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {MODEL_NAME}@{args.version} in {args.dst}")
    print(json.dumps({k: info[k] for k in ("generation", "test_metrics", "files_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Фикстура: крошечная T5-модель того же класса, что ruT5-base, со своим SentencePiece-токенизатором.
Позволяет проверить весь путь (prepare → load → hash → predict) без скачивания 900-МБ весов."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = [
    "Блюдо: Бешбармак Категория: Горячие блюда > Блюда из мяса Ингредиенты: конина, баранина, тесто, лук",
    "Нежный пирог с яблоками и корицей к чаю.", "Сытный суп из говядины с овощами и зеленью.",
    "Салат из свежих овощей с оливковым маслом.", "Ароматная выпечка с творогом и изюмом.",
] * 40


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    import sentencepiece as spm
    import torch
    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

    work = tmp_path_factory.mktemp("tiny")
    (work / "corpus.txt").write_text("\n".join(CORPUS), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(work / "corpus.txt"), model_prefix=str(work / "spiece"),
                                   vocab_size=60, model_type="char", pad_id=0, eos_id=1, unk_id=2,
                                   bos_id=-1, character_coverage=1.0)
    src = work / "rut5_menu_desc"
    tok = T5Tokenizer(vocab_file=str(work / "spiece.model"), extra_ids=0, legacy=False)
    tok.save_pretrained(src)
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=len(tok), d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=2,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    T5ForConditionalGeneration(cfg).save_pretrained(src)

    results = work / "final_results.csv"
    results.write_text("model,chrF++,chrF++_b1,ROUGE-L,BERTScore_F1,ingredient_mention_%,health_claim_%,avg_chars\n"
                       "ruT5-base + tuned decoding,17.69,20.12,17.76,53.06,75.7,23.3,71.0\n", encoding="utf-8")
    dst = work / "models" / "rut5-menu-desc" / "1.0.0"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_final_model.py"), "--src", str(src),
                    "--dst", str(dst), "--results", str(results)], check=True, capture_output=True)
    return dst


@pytest.fixture(scope="session")
def generator(tiny_model_dir):
    from menu_desc import MenuDescriptionGenerator
    return MenuDescriptionGenerator.from_dir(tiny_model_dir, device="cpu")


@pytest.fixture
def version_info(tiny_model_dir) -> dict:
    return json.loads((tiny_model_dir / "model_version.json").read_text(encoding="utf-8"))

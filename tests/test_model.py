import json
import shutil

import pytest

from menu_desc import DescriptionOutput, InputValidationError, MenuDescriptionGenerator, ModelIntegrityError
from conftest import ROOT

DISH = {"title": "Бешбармак", "category": ["Горячие блюда", "Блюда из мяса"],
        "ingredients": ["конина", "баранина", "тесто", "лук репчатый", "бульон"]}


def test_version_file_pins_model(version_info):
    assert version_info["name"] == "rut5-menu-desc" and version_info["version"] == "1.0.0"
    assert version_info["generation"] == {"num_beams": 4, "no_repeat_ngram_size": 2, "length_penalty": 2.0,
                                          "min_new_tokens": 15, "max_new_tokens": 80, "early_stopping": True,
                                          "do_sample": False}
    assert len(version_info["files_sha256"]["model.safetensors"]) == 64
    assert version_info["test_metrics"]["BERTScore_F1"] == 53.06


def test_generation_config_saved_with_weights(tiny_model_dir):
    gen_cfg = json.loads((tiny_model_dir / "generation_config.json").read_text())
    assert gen_cfg["num_beams"] == 4 and gen_cfg["min_new_tokens"] == 15


def test_predict_returns_contract(generator):
    out = generator.predict(DISH)
    assert isinstance(out, DescriptionOutput)
    assert out.model_version == "rut5-menu-desc@1.0.0"
    assert isinstance(out.description, str)
    assert 0 < out.input_tokens <= 128 and out.latency_ms >= 0


def test_predict_is_deterministic(generator):
    assert generator.predict(DISH).description == generator.predict(DISH).description


def test_batch_equals_single(generator):
    other = {"title": "Боул с лососем", "category": "Горячие блюда > Блюда из рыбы",
             "ingredients": "лосось, рис, авокадо"}
    batch = generator.predict_batch([DISH, other])
    assert [o.description for o in batch] == [generator.predict(DISH).description,
                                              generator.predict(other).description]


def test_string_and_list_inputs_equivalent(generator):
    as_str = {**DISH, "category": "Горячие блюда > Блюда из мяса", "ingredients": ", ".join(DISH["ingredients"])}
    assert generator.prepare(as_str) == generator.prepare(DISH)


def test_long_input_truncated_and_flagged(generator):
    dish = {"title": "Ассорти", "ingredients": [f"очень длинный ингредиент номер {i}" for i in range(30)]}
    out = generator.predict(dish)
    assert out.input_tokens == 128 and "input_truncated" in out.warnings


def test_invalid_input_raises(generator):
    with pytest.raises(InputValidationError):
        generator.predict({"title": "Суп", "ingredients": []})
    with pytest.raises(Exception):              # pydantic: обязательное поле отсутствует
        generator.predict({"ingredients": ["рис"]})


def test_tampered_weights_rejected(tiny_model_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(tiny_model_dir, bad)
    with open(bad / "model.safetensors", "ab") as f:
        f.write(b"0")
    with pytest.raises(ModelIntegrityError):
        MenuDescriptionGenerator.from_dir(bad)


def test_int8_quantized_model_runs(tiny_model_dir):
    gen = MenuDescriptionGenerator.from_dir(tiny_model_dir, device="cpu", quantize_int8=True)
    assert isinstance(gen.predict(DISH).description, str)


def test_prepare_refuses_to_overwrite_version(tiny_model_dir):
    import subprocess, sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_final_model.py"
    r = subprocess.run([sys.executable, str(script), "--src", str(tiny_model_dir), "--dst", str(tiny_model_dir)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "не перезаписываются" in r.stderr


def test_prepare_reads_best_decoding(tmp_path):
    import importlib.util, json
    spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts" / "prepare_final_model.py")
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    p = tmp_path / "experiments.json"
    p.write_text(json.dumps({"best": {"t5_decoding": {"num_beams": "4", "length_penalty": "2.0",
                                                      "no_repeat_ngram_size": 2, "min_new_tokens": 15}}}))
    assert prep.read_best_decoding(p) == {"num_beams": 4, "length_penalty": 2.0,
                                          "no_repeat_ngram_size": 2, "min_new_tokens": 15}
    assert prep.read_best_decoding(tmp_path / "missing.json") == {}

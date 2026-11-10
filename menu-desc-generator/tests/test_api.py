"""Тесты HTTP API на крошечной T5 (фикстура tiny_model_dir из conftest.py)."""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

DISH = {"title": "Бешбармак", "category": "Горячие блюда > Блюда из мяса",
        "ingredients": "конина, баранина, тесто, лук"}


def make_settings(model_dir, **kw):
    base = dict(model_dir=model_dir, model_repo=None, model_revision="main", hf_token=None,
                quantize_int8=False, num_threads=1, max_batch=3)
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="module")
def client(tiny_model_dir):
    app = create_app(make_settings(tiny_model_dir), load_in_background=False)
    with TestClient(app) as c:
        yield c


def test_health_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_state"] == "ready"


def test_version_exposes_pinned_model(client, version_info):
    v = client.get("/version").json()
    assert v["model_version"] == f"{version_info['name']}@{version_info['version']}"
    assert v["weights_sha256"] == version_info["files_sha256"]["model.safetensors"]
    assert v["generation"]["num_beams"] == 4


def test_predict_contract(client, version_info):
    r = client.post("/predict", json=DISH)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"description", "model_version", "warnings", "input_tokens", "latency_ms"}
    assert body["model_version"].endswith("@" + version_info["version"])
    assert "x-request-id" in r.headers


def test_predict_matches_module(client, generator):
    assert client.post("/predict", json=DISH).json()["description"] == generator.predict(DISH).description


@pytest.mark.parametrize("payload", [
    {"title": "Суп", "ingredients": []},                 # нет ингредиентов
    {"title": "12345", "ingredients": "рис"},             # название без букв
    {"ingredients": "рис"},                               # нет названия
    {"title": "Суп", "ingredients": 5},                   # неверный тип
])
def test_invalid_input_returns_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_batch(client):
    r = client.post("/predict/batch", json={"dishes": [DISH, {**DISH, "title": "Плов"}]})
    assert r.status_code == 200 and len(r.json()["results"]) == 2


def test_batch_limit(client):
    r = client.post("/predict/batch", json={"dishes": [DISH] * 4})
    assert r.status_code == 422


def test_metrics_count_requests_and_errors(client):
    before = client.get("/metrics").json()
    client.post("/predict", json=DISH)
    client.post("/predict", json={"title": "Суп", "ingredients": []})
    after = client.get("/metrics").json()
    assert after["requests_by_endpoint"]["/predict"] == before["requests_by_endpoint"].get("/predict", 0) + 2
    assert after["responses_by_status"]["422"] >= 1
    assert after["descriptions_total"] == before["descriptions_total"] + 1
    assert after["latency_ms"]["p50"] is not None


def test_metrics_prometheus(client):
    r = client.get("/metrics", params={"format": "prometheus"})
    assert r.status_code == 200 and "menu_desc_requests_total" in r.text


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Генератор описаний блюд" in r.text


def test_missing_model_reports_error(tmp_path):
    app = create_app(make_settings(tmp_path / "nothing"), load_in_background=False)
    with TestClient(app) as c:
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["model_state"] == "error"
        assert c.post("/predict", json=DISH).status_code == 503


def test_model_downloaded_from_hub_when_missing(tiny_model_dir, tmp_path, monkeypatch):
    """Если локальной версии нет, сервис скачивает её из MODEL_REPO с зафиксированным тегом."""
    import shutil
    import huggingface_hub
    calls = {}

    def fake_snapshot_download(repo_id, revision, local_dir, token=None):
        calls.update(repo_id=repo_id, revision=revision)
        shutil.copytree(tiny_model_dir, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    settings = make_settings(tmp_path / "dl", model_repo="user/rut5-menu-desc", model_revision="v1.0.0")
    with TestClient(create_app(settings, load_in_background=False)) as c:
        assert c.get("/health").json()["model_state"] == "ready"
        assert c.post("/predict", json=DISH).status_code == 200
    assert calls == {"repo_id": "user/rut5-menu-desc", "revision": "v1.0.0"}


def test_int8_service(tiny_model_dir):
    with TestClient(create_app(make_settings(tiny_model_dir, quantize_int8=True), load_in_background=False)) as c:
        assert c.get("/version").json()["quantization"] == "dynamic-int8"
        assert c.post("/predict", json=DISH).status_code == 200

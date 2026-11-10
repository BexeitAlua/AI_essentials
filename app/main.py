"""HTTP API и веб-интерфейс сервиса генерации описаний блюд.

Эндпоинты:
    GET  /               веб-форма
    POST /predict        одно блюдо -> описание
    POST /predict/batch  до MAX_BATCH блюд за запрос
    GET  /health         статус сервиса и модели (loading / ready / error)
    GET  /version        версия модели, параметры генерации, хэш весов
    GET  /metrics        мониторинг: JSON или Prometheus (?format=prometheus)
    GET  /docs           Swagger UI (автоматически от FastAPI)

Запуск локально:  MODEL_DIR=models/rut5-menu-desc/1.0.0 uvicorn app.main:app --port 7860
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

from menu_desc import DescriptionOutput, DishInput, InputValidationError, MenuDescriptionGenerator
from menu_desc.model import VERSION_FILE

from .config import Settings, load_settings
from .monitoring import Metrics, log_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("menu_desc.api")
STATIC = Path(__file__).parent / "static"
API_VERSION = "1.0.0"


class ModelService:
    """Загружает модель в фоне, чтобы порт открылся сразу и /health показывал прогресс."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.generator: MenuDescriptionGenerator | None = None
        self.state = "loading"
        self.error: str | None = None
        self.load_seconds: float | None = None
        self._lock = threading.Lock()     # генерация на CPU: один запрос к модели за раз

    def _resolve_model_dir(self) -> Path:
        s = self.settings
        if (s.model_dir / VERSION_FILE).exists():
            return s.model_dir
        if not s.model_repo:
            raise FileNotFoundError(f"Нет {s.model_dir / VERSION_FILE} и не задан MODEL_REPO")
        from huggingface_hub import snapshot_download
        logger.info("Downloading %s@%s -> %s", s.model_repo, s.model_revision, s.model_dir)
        snapshot_download(repo_id=s.model_repo, revision=s.model_revision, local_dir=s.model_dir,
                          token=s.hf_token)
        return s.model_dir

    def load(self) -> None:
        t0 = time.time()
        try:
            model_dir = self._resolve_model_dir()
            self.generator = MenuDescriptionGenerator.from_dir(
                model_dir, device="cpu", quantize_int8=self.settings.quantize_int8,
                num_threads=self.settings.num_threads)
            self.load_seconds = round(time.time() - t0, 1)
            self.state = "ready"
            log_event(event="model_loaded", **self.generator.describe(), load_seconds=self.load_seconds,
                      int8=self.settings.quantize_int8)
        except Exception as e:                       # сервис остаётся живым и сообщает причину в /health
            self.state, self.error = "error", f"{type(e).__name__}: {e}"
            logger.exception("Model loading failed")

    def start(self, background: bool = True) -> None:
        if background:
            threading.Thread(target=self.load, daemon=True, name="model-loader").start()
        else:
            self.load()

    def predict(self, dishes: list) -> list[DescriptionOutput]:
        if self.state != "ready":
            raise HTTPException(503, detail=f"Модель не готова: {self.state}"
                                            + (f" ({self.error})" if self.error else ""))
        with self._lock:
            return self.generator.predict_batch(dishes)


class BatchRequest(BaseModel):
    dishes: list[DishInput] = Field(..., min_length=1, description="Список блюд")


class BatchResponse(BaseModel):
    results: list[DescriptionOutput]


def create_app(settings: Settings | None = None, load_in_background: bool = True) -> FastAPI:
    settings = settings or load_settings()
    service = ModelService(settings)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start(background=load_in_background)
        yield

    app = FastAPI(title="Menu description generator", version=API_VERSION, lifespan=lifespan,
                  description="Генерация описаний блюд для карточек меню (ruT5-base, fine-tuned).")
    app.state.service, app.state.metrics = service, metrics

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            ms = (time.perf_counter() - t0) * 1000
            path = request.url.path
            if path not in ("/favicon.ico",):
                metrics.observe_request(path, status, ms)
            if path.startswith("/predict"):
                log_event(event="request", request_id=request_id, path=path, status=status,
                          latency_ms=round(ms, 1))

    @app.exception_handler(InputValidationError)
    async def input_error(request: Request, exc: InputValidationError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/predict", response_model=DescriptionOutput, summary="Описание для одного блюда")
    def predict(dish: DishInput):
        out = service.predict([dish])
        metrics.observe_predictions(out)
        log_event(event="prediction", title=dish.title, warnings=out[0].warnings,
                  model_latency_ms=out[0].latency_ms, chars=len(out[0].description))
        return out[0]

    @app.post("/predict/batch", response_model=BatchResponse, summary=f"Описания для 1–{settings.max_batch} блюд")
    def predict_batch(req: BatchRequest):
        if len(req.dishes) > settings.max_batch:
            raise HTTPException(422, detail=f"Не больше {settings.max_batch} блюд за запрос")
        outs = service.predict(req.dishes)
        metrics.observe_predictions(outs)
        return {"results": outs}

    @app.get("/health", summary="Статус сервиса")
    def health():
        body = {"status": "ok" if service.state == "ready" else service.state, "model_state": service.state,
                "uptime_s": metrics.snapshot()["uptime_s"]}
        if service.error:
            body["error"] = service.error
        if service.load_seconds is not None:
            body["model_load_seconds"] = service.load_seconds
        # 200 пока модель грузится (сервис жив), 503 если загрузка упала
        return JSONResponse(status_code=503 if service.state == "error" else 200, content=body)

    @app.get("/version", summary="Версия модели и API")
    def version():
        if service.generator is None:
            raise HTTPException(503, detail=f"Модель не готова: {service.state}")
        g = service.generator
        return {"api_version": API_VERSION, **g.describe(), "quantization": "dynamic-int8" if
                settings.quantize_int8 else "none", "test_metrics": g.info.get("test_metrics", {}),
                "input": g.info.get("input", {})}

    @app.get("/metrics", summary="Метрики мониторинга")
    def get_metrics(format: str = "json"):
        if format == "prometheus":
            return PlainTextResponse(metrics.prometheus())
        return {**metrics.snapshot(), "model_state": service.state}

    return app


app = create_app()

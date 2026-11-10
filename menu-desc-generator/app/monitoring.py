"""Мониторинг в памяти процесса: счётчики запросов, ошибок, предупреждений модели и задержки.

Отдаётся эндпоинтом /metrics в двух форматах: JSON (для веб-интерфейса) и Prometheus text
(для внешнего мониторинга). Каждый запрос дополнительно пишется в лог одной JSON-строкой —
их видно во вкладке Logs у Hugging Face Space.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque

logger = logging.getLogger("menu_desc.requests")


class Metrics:
    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = Counter()          # endpoint -> число запросов
        self.status = Counter()            # HTTP-код -> число ответов (все эндпоинты)
        self.predict_status = Counter()    # HTTP-код -> число ответов /predict* (для доли ошибок)
        self.warnings = Counter()          # флаг модели -> сколько раз
        self.descriptions = 0              # сколько описаний сгенерировано
        self.latency_ms = deque(maxlen=window)        # полное время ответа /predict
        self.model_latency_ms = deque(maxlen=window)  # время генерации моделью

    def observe_request(self, endpoint: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self.requests[endpoint] += 1
            self.status[str(status)] += 1
            if endpoint.startswith("/predict"):
                self.predict_status[str(status)] += 1
                if status == 200:
                    self.latency_ms.append(latency_ms)

    def observe_predictions(self, outputs) -> None:
        with self._lock:
            for out in outputs:
                self.descriptions += 1
                self.model_latency_ms.append(out.latency_ms)
                self.warnings.update(out.warnings)

    @staticmethod
    def _pct(values, q: float) -> float | None:
        if not values:
            return None
        s = sorted(values)
        return round(s[min(len(s) - 1, int(q * len(s)))], 1)

    def snapshot(self) -> dict:
        with self._lock:
            total = sum(self.status.values())
            predicts = sum(self.predict_status.values())
            client_err = sum(n for c, n in self.predict_status.items() if c.startswith("4"))
            server_err = sum(n for c, n in self.predict_status.items() if c.startswith("5"))
            lat = list(self.latency_ms)
            return {
                "uptime_s": round(time.time() - self.started_at),
                "requests_total": total,
                "requests_by_endpoint": dict(self.requests),
                "responses_by_status": dict(self.status),
                "predict_requests": predicts,
                # 4xx — некорректный ввод пользователя, 5xx — сбой сервиса или модель не готова
                "error_rate_%": round(100 * (client_err + server_err) / predicts, 1) if predicts else 0.0,
                "server_error_rate_%": round(100 * server_err / predicts, 1) if predicts else 0.0,
                "descriptions_total": self.descriptions,
                "warnings": dict(self.warnings),
                "warning_rate_%": {k: round(100 * v / self.descriptions, 1)
                                   for k, v in self.warnings.items()} if self.descriptions else {},
                "latency_ms": {"p50": self._pct(lat, 0.5), "p95": self._pct(lat, 0.95),
                               "max": round(max(lat), 1) if lat else None, "window": len(lat)},
            }

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = ["# TYPE menu_desc_requests_total counter"]
        for code, n in snap["responses_by_status"].items():
            lines.append(f'menu_desc_requests_total{{status="{code}"}} {n}')
        lines.append("# TYPE menu_desc_descriptions_total counter")
        lines.append(f"menu_desc_descriptions_total {snap['descriptions_total']}")
        lines.append("# TYPE menu_desc_warnings_total counter")
        for flag, n in snap["warnings"].items():
            lines.append(f'menu_desc_warnings_total{{flag="{flag}"}} {n}')
        lines.append("# TYPE menu_desc_latency_ms gauge")
        for q in ("p50", "p95"):
            if snap["latency_ms"][q] is not None:
                lines.append(f'menu_desc_latency_ms{{quantile="{q}"}} {snap["latency_ms"][q]}')
        lines.append(f"menu_desc_uptime_seconds {snap['uptime_s']}")
        return "\n".join(lines) + "\n"


def log_event(**fields) -> None:
    """Одна строка JSON на событие: удобно читать в логах и парсить."""
    logger.info(json.dumps(fields, ensure_ascii=False))

"""Контракт входа и выхода модели (используется и модулем inference, и будущим API)."""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field


class DishInput(BaseModel):
    title: str = Field(..., description="Название блюда", examples=["Бешбармак"])
    category: Optional[Union[str, list[str]]] = Field(
        None, description="Категория: 'Уровень 1 > Уровень 2' или список уровней (берутся 2 верхних)",
        examples=[["Горячие блюда", "Блюда из мяса"]])
    ingredients: Union[str, list[str]] = Field(
        ..., description="Ингредиенты: список или строка через запятую (1–30 шт.)",
        examples=[["конина", "баранина", "тесто", "лук репчатый", "бульон"]])


class DescriptionOutput(BaseModel):
    description: str = Field(..., description="Сгенерированное описание блюда (русский язык)")
    model_version: str = Field(..., description="Идентификатор версии модели, например 'rut5-menu-desc@1.0.0'")
    warnings: list[str] = Field(default_factory=list, description=(
        "Флаги проверок: health_claim, first_person, no_ingredient_mentioned, empty_output, input_truncated"))
    input_tokens: int = Field(..., description="Длина входа модели в токенах (после обрезки до 128)")
    latency_ms: float = Field(..., description="Время генерации, мс")

"""Предобработка входа и пост-проверки выхода.

Формат входной строки модели ДОЛЖЕН совпадать с обучением (Midterm, раздел 4, make_source):
    "Блюдо: {title}\\nКатегория: {cat1 > cat2}\\nИнгредиенты: {i1, i2, ...}"
Любое отличие (другой разделитель, порядок полей) ухудшает качество, поэтому функция
build_source — единственное место, где этот формат собирается.
"""
from __future__ import annotations

import re
import unicodedata

# Ограничения входа (подобраны по обучающим данным: названия ≤ 100 символов,
# 95% рецептов имеют ≤ 20 ингредиентов, вход модели обрезается до 128 токенов)
MAX_TITLE_CHARS = 100
MAX_INGREDIENTS = 30
MAX_INGREDIENT_CHARS = 60
MAX_CATEGORY_LEVELS = 2          # при обучении использовались 2 верхних уровня «крошек»
NO_CATEGORY = "—"                # заглушка из обучения, если категория неизвестна

# Те же регулярные выражения, что в продуктовых проверках ноутбука (раздел 6)
HEALTH_CLAIM = re.compile(
    r"\b(полезн\w*|диетич\w*|низкокалор\w*|здоров\w*|витамин\w*|похуд\w*|самый|самая|самое|лучш\w*|идеальн\w*)\b",
    re.IGNORECASE)
FIRST_PERSON = re.compile(r"\b(я|мне|меня|мой|моя|моё|мое|мои|мы|нас|наш|наша|наши|муж\w*)\b", re.IGNORECASE)
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_SPACES = re.compile(r"\s+")


class InputValidationError(ValueError):
    """Вход не может быть обработан моделью. Сообщение пригодно для показа пользователю."""


def clean_text(value: object) -> str:
    """Unicode-нормализация, удаление управляющих символов, схлопывание пробелов."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    return _SPACES.sub(" ", text).strip()


def _split(value: object, sep_pattern: str) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else re.split(sep_pattern, str(value))
    return [clean_text(x) for x in items if clean_text(x)]


def normalize_title(title: object) -> str:
    title = clean_text(title)
    if not title:
        raise InputValidationError("Поле 'title' обязательно и не может быть пустым.")
    if not _LETTER.search(title):
        raise InputValidationError("Название блюда должно содержать хотя бы одну букву.")
    if len(title) > MAX_TITLE_CHARS:
        raise InputValidationError(f"Название блюда длиннее {MAX_TITLE_CHARS} символов.")
    return title


def normalize_category(category: object) -> list[str]:
    """Категория: список уровней или строка вида 'Горячие блюда > Блюда из мяса'."""
    levels = _split(category, r"\s*>\s*|\s*/\s*")
    return levels[:MAX_CATEGORY_LEVELS]


def normalize_ingredients(ingredients: object) -> list[str]:
    """Ингредиенты: список или строка через запятую/точку с запятой/перенос строки.
    Пустые элементы удаляются, дубликаты (без учёта регистра) схлопываются, порядок сохраняется."""
    items = _split(ingredients, r"[,;\n]")
    seen, result = set(), []
    for item in items:
        if len(item) > MAX_INGREDIENT_CHARS:
            raise InputValidationError(f"Ингредиент длиннее {MAX_INGREDIENT_CHARS} символов: '{item[:30]}…'.")
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    if not result:
        raise InputValidationError("Нужен хотя бы один ингредиент.")
    if len(result) > MAX_INGREDIENTS:
        raise InputValidationError(f"Слишком много ингредиентов: {len(result)} (максимум {MAX_INGREDIENTS}).")
    return result


def build_source(title: str, category_levels: list[str], ingredients: list[str]) -> str:
    """Строка входа модели — точная копия make_source() из обучения."""
    category = " > ".join(category_levels) if category_levels else NO_CATEGORY
    return f"Блюдо: {title}\nКатегория: {category}\nИнгредиенты: {', '.join(ingredients)}"


def _stems(text: str, k: int = 5) -> set[str]:
    # та же грубая «лемматизация», что в метрике ingredient_mention_% ноутбука
    return {w[:k] for w in re.findall(r"\w+", text.lower()) if len(w) >= 4}


def output_warnings(description: str, ingredients: list[str]) -> list[str]:
    """Продуктовые проверки сгенерированного текста. Текст не меняется — только флаги для UI/мониторинга."""
    warnings = []
    if not description:
        warnings.append("empty_output")
        return warnings
    if HEALTH_CLAIM.search(description):
        warnings.append("health_claim")             # «полезный», «лучший»… — требует модерации
    if FIRST_PERSON.search(description):
        warnings.append("first_person")
    desc_stems = _stems(description)
    if not any(_stems(i) & desc_stems for i in ingredients):
        warnings.append("no_ingredient_mentioned")
    return warnings

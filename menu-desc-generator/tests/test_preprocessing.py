import pytest

from menu_desc.preprocessing import (InputValidationError, build_source, normalize_category,
                                     normalize_ingredients, normalize_title, output_warnings)


def test_build_source_matches_training_format():
    # точная копия формата make_source() из ноутбука Midterm
    src = build_source("Бешбармак", ["Горячие блюда", "Блюда из мяса"], ["конина", "тесто"])
    assert src == "Блюдо: Бешбармак\nКатегория: Горячие блюда > Блюда из мяса\nИнгредиенты: конина, тесто"


def test_missing_category_uses_training_placeholder():
    assert "Категория: —\n" in build_source("Суп", [], ["вода"])


@pytest.mark.parametrize("raw, expected", [
    ("Горячие блюда > Блюда из мяса > Говядина", ["Горячие блюда", "Блюда из мяса"]),
    (["Выпечка", " Пироги "], ["Выпечка", "Пироги"]),
    (None, []),
    ("", []),
])
def test_category_normalization(raw, expected):
    assert normalize_category(raw) == expected


def test_ingredients_string_dedup_and_cleanup():
    assert normalize_ingredients(" лук ,Лук; морковь\n\n  картофель ,, ") == ["лук", "морковь", "картофель"]


@pytest.mark.parametrize("bad", [None, "", " , ; ", []])
def test_empty_ingredients_rejected(bad):
    with pytest.raises(InputValidationError):
        normalize_ingredients(bad)


def test_too_many_ingredients_rejected():
    with pytest.raises(InputValidationError):
        normalize_ingredients([f"ингредиент {i}" for i in range(31)])


@pytest.mark.parametrize("bad", ["", "   ", "12345", "!!!", "а" * 101])
def test_bad_titles_rejected(bad):
    with pytest.raises(InputValidationError):
        normalize_title(bad)


def test_title_control_chars_and_spaces_cleaned():
    assert normalize_title("Борщ\t\x00  украинский\n") == "Борщ украинский"


def test_output_warnings():
    assert output_warnings("Сочный бешбармак из конины с тестом.", ["конина", "тесто"]) == []
    assert "health_claim" in output_warnings("Полезный суп из конины.", ["конина"])
    assert "no_ingredient_mentioned" in output_warnings("Очень вкусное блюдо.", ["лосось", "рис"])
    assert output_warnings("", ["рис"]) == ["empty_output"]

---
title: Menu Description Generator
emoji: 🍽️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Описания блюд для меню (ruT5-base, fine-tuned)
---

# Генератор описаний блюд

Прототип интеллектуальной системы: по названию, категории и ингредиентам блюда модель
`rut5-menu-desc@1.0.0` (ruT5-base, дообученная на рецептах «Повар.ру») пишет короткое описание для карточки меню.

- **Веб-интерфейс:** главная страница Space
- **API:** `POST /predict`, `POST /predict/batch`, документация — `/docs`
- **Мониторинг:** `/health`, `/version`, `/metrics` (JSON или `?format=prometheus`)

Модель загружается из репозитория модели при старте, SHA-256 весов сверяется с `model_version.json`,
затем модель квантуется в INT8 (CPU: ≈ 0.9 с на запрос, качество как у FP32).

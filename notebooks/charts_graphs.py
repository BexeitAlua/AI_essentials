from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#CFCDC6", "axes.labelcolor": "#4A4A47", "xtick.color": "#4A4A47", "ytick.color": "#4A4A47",
    "figure.facecolor": "#FBFBF8", "axes.facecolor": "#FBFBF8", "savefig.facecolor": "#FBFBF8",
})
BLUE, ORANGE, GREEN, GREY, LGREY, PINK, YELLOW = ("#2D7DD2", "#E8683A", "#1DAE7A", "#A5A49E",
                                                  "#D9D7D1", "#E07AA5", "#E9A23B")
INK, MUTED = "#151515", "#5C5B57"

FINAL = "ruT5-base + tuned decoding"          # строка финальной модели в final_results.csv
TUNED = "ruT5-base tuned"                     # кандидат с подбором lr/batch
# имя в final_results.csv -> (подпись на графике, цвет группы)
MODELS = {
    "Template (no ML)": ("Шаблон (правила)", GREY),
    "kNN-TFIDF default": ("kNN TF-IDF, default", LGREY),
    "kNN-TFIDF tuned": ("kNN TF-IDF, подбор", BLUE),
    "Qwen2.5-0.5B zero-shot": ("Qwen2.5-0.5B zero-shot", YELLOW),
    "Qwen2.5-0.5B + LoRA": ("Qwen2.5-0.5B + LoRA", PINK),
    "ruT5-base fine-tuned": ("ruT5-base fine-tuned", GREEN),
    TUNED: ("ruT5 + подбор lr/batch", GREEN),
    FINAL: ("ruT5 + подбор декодирования ★", ORANGE),
}
# имена в diagnostics.csv -> подпись
DIAG_NAMES = {
    "Qwen zero-shot": "Qwen zero-shot", "kNN tuned": "kNN, подбор", "kNN default": "kNN, default",
    "Template": "Шаблон", "ruT5 tuned": "ruT5 + lr/batch", "ruT5 + tuned decoding": "ruT5 + декодирование ★",
    "Qwen + LoRA": "Qwen + LoRA", "ruT5 fine-tuned": "ruT5 fine-tuned",
}


def head(fig, title: str, sub: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=24, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.885, sub, fontsize=15, color=MUTED, va="top")


def results_chart(res: pd.DataFrame, out: Path) -> None:
    rows = res.loc[[m for m in MODELS if m in res.index]]
    labels = [MODELS[m][0] for m in rows.index]
    cols = [MODELS[m][1] for m in rows.index]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.74, bottom=0.17, wspace=0.12)
    head(fig, f"Сравнение {len(rows)} моделей на одном test",
         "Одинаковый split, метрики и вход для всех · ★ — финальная модель · BERTScore: ось начинается с 44")
    y = np.arange(len(rows))[::-1]
    for ax, col, name, lo in zip(axes, ["chrF++_b1", "ROUGE-L", "BERTScore_F1"],
                                 ["chrF++ (β=1)", "ROUGE-L", "BERTScore F1"], [0, 0, 44]):
        vals = rows[col].to_numpy(float)
        ax.barh(y, vals - lo, left=lo, color=cols, height=0.68)
        for yi, v in zip(y, vals):
            ax.text(v + (0.25 if lo == 0 else 0.12), yi, f"{v:.2f}", va="center", fontsize=13.5, color=INK)
        ax.set_title(name, loc="left", fontsize=16, color=INK, pad=10)
        ax.set_xlim(lo, vals.max() * 1.13 if lo == 0 else max(55.5, vals.max() + 2))
        ax.tick_params(axis="x", labelsize=12)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=14)
    for t in axes[0].get_yticklabels():
        if "★" in t.get_text():
            t.set_fontweight("bold")
    leg = [Patch(color=GREY, label="правила"), Patch(color=BLUE, label="классический ML"),
           Patch(color=YELLOW, label="LLM без обучения"), Patch(color=PINK, label="LLM + LoRA"),
           Patch(color=GREEN, label="transfer learning + fine-tuning"), Patch(color=ORANGE, label="финальная модель")]
    fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=6, frameon=False, fontsize=13)
    fig.savefig(out / "results.png", dpi=130)
    plt.close(fig)


def beta_chart(diag: pd.DataFrame, out: Path) -> None:
    ref_chars = diag.loc["Reference (test targets)", "avg_chars"] if "Reference (test targets)" in diag.index else None
    d = diag.loc[[m for m in DIAG_NAMES if m in diag.index]].sort_values("avg_chars", ascending=False)
    b2 = d["chrF++ beta=2 (recall x2)"].to_numpy(float)
    b1 = d["chrF++ beta=1 (balanced)"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(16, 7.2))
    fig.subplots_adjust(left=0.2, right=0.86, top=0.75, bottom=0.12)
    sub = "β=2 считает полноту вдвое важнее точности"
    if ref_chars is not None:
        sub += f" · эталонные описания — {ref_chars:.0f} символов в среднем"
    head(fig, "Почему chrF++ с β=1: стандартный β=2 награждает длину", sub)
    yy = np.arange(len(d))[::-1]
    lo, hi = min(b1.min(), b2.min()) - 0.6, max(b1.max(), b2.max()) + 0.9
    for yi, x1, x2, chars in zip(yy, b1, b2, d["avg_chars"]):
        ax.plot([x1, x2], [yi, yi], color=LGREY, lw=4, zorder=1)
        ax.scatter(x1, yi, s=160, color=BLUE, zorder=3)
        ax.scatter(x2, yi, s=160, color=ORANGE, zorder=3)
        ax.text(hi + 0.3, yi, f"{chars:.0f} симв.", va="center", fontsize=14, color=MUTED)
    labels = [DIAG_NAMES[m] for m in d.index]
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=14)
    for t in ax.get_yticklabels():
        if "★" in t.get_text():
            t.set_fontweight("bold")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("chrF++ на test")
    ax.scatter([], [], s=120, color=BLUE, label="β = 1 (точность = полнота)")
    ax.scatter([], [], s=120, color=ORANGE, label="β = 2 (стандарт)")
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    ax.annotate("самые длинные ответы", xy=(b2[0], yy[0]), xytext=(b2[0] + 0.1, yy[2]),
                fontsize=13, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.savefig(out / "beta.png", dpi=130)
    plt.close(fig)


def waterfall_chart(res: pd.DataFrame, out: Path) -> None:
    path = ["Template (no ML)", "ruT5-base fine-tuned", FINAL, TUNED]
    steps = ["шаблон\n(без ML)", "+ fine-tuning\nruT5", "+ подбор\nдекодирования", "+ подбор\nlr/batch"]

    def wf(ax, col, title):
        vals = res.loc[path, col].to_numpy(float)
        deltas = np.diff(vals)
        ax.bar(0, vals[0], color=GREY, width=0.6)
        ax.text(0, vals[0] + 0.25, f"{vals[0]:.2f}", ha="center", fontsize=14)
        for i, dl in enumerate(deltas, 1):
            b = vals[i - 1]
            c = ORANGE if dl == deltas.max() else (GREEN if dl > 0 else PINK)
            ax.bar(i, dl, bottom=b, color=c, width=0.6)
            ax.plot([i - 1.3, i + 0.3], [b, b], ls=":", color=MUTED, lw=1)
            ax.text(i, max(b, b + dl) + 0.25, f"{dl:+.2f}", ha="center", fontsize=14, color=INK)
        ax.set_xticks(range(len(path)))
        ax.set_xticklabels(steps, fontsize=13)
        ax.set_title(title, loc="left", fontsize=16, color=INK, pad=8)
        ax.set_ylim(0, vals.max() * 1.18)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.16, wspace=0.15)
    head(fig, "Откуда прирост: каждый шаг отдельно",
         "Test · оранжевый — самый большой вклад · fine-tuning даёт смысл (ROUGE-L), декодирование — длину и chrF++")
    wf(axes[0], "ROUGE-L", "ROUGE-L")
    wf(axes[1], "chrF++_b1", "chrF++ (β=1)")
    fig.savefig(out / "waterfall.png", dpi=130)
    plt.close(fig)


def tuning_chart(dec: pd.DataFrame, lr: pd.DataFrame, base_cfg: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.13, wspace=0.28)
    head(fig, "Подбор гиперпараметров на validation",
         f"Слева: {len(dec)} настроек генерации (обученная модель та же) · справа: {len(lr)} настроек обучения")
    ax = axes[0]
    for _, r in dec.iterrows():
        ax.scatter(r["avg_chars"], r["chrF++_b1"], s=210 if r["num_beams"] > 1 else 110,
                   color=ORANGE if r["min_new_tokens"] > 0 else GREY,
                   marker="o" if r["num_beams"] > 1 else "s", edgecolor="white", zorder=3)
    best = dec.loc[dec["select"].idxmax()]
    ax.scatter(best["avg_chars"], best["chrF++_b1"], s=650, facecolor="none", edgecolor=INK, lw=2, zorder=4)
    ax.annotate(f"выбрано: beams {best['num_beams']:.0f}, no_repeat {best['no_repeat_ngram_size']:.0f},\n"
                f"length_penalty {best['length_penalty']}, min_new_tokens {best['min_new_tokens']:.0f}",
                xy=(best["avg_chars"], best["chrF++_b1"]),
                xytext=(dec["avg_chars"].min() + 2, best["chrF++_b1"] + 0.1),
                fontsize=12.5, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("средняя длина ответа, символов")
    ax.set_ylabel("chrF++ (β=1), validation")
    ax.set_ylim(dec["chrF++_b1"].min() - 0.4, dec["chrF++_b1"].max() + 0.55)
    ax.set_title("Декодирование", loc="left", fontsize=16, color=INK, pad=8)
    ax.scatter([], [], color=ORANGE, s=100, label="с min_new_tokens")
    ax.scatter([], [], color=GREY, s=100, label="без минимума")
    ax.legend(loc="lower right", frameon=False, fontsize=12.5)

    ax = axes[1]
    lr = lr.sort_values("val_loss").reset_index(drop=True)
    names = [f"{r.learning_rate:g} · {r.batch_size:.0f}" for r in lr.itertuples()]
    is_base = [(abs(r.learning_rate - base_cfg.get("learning_rate", -1)) < 1e-12
                and r.batch_size == base_cfg.get("batch_size", -1)) for r in lr.itertuples()]
    colors = [GREEN if i == 0 else (GREY if b else BLUE) for i, b in enumerate(is_base)]
    yy = np.arange(len(lr))[::-1]
    ax.barh(yy, lr["val_loss"], color=colors, height=0.65)
    for yi, v in zip(yy, lr["val_loss"]):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=13)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=13.5)
    lo = lr["val_loss"].min() - 0.07
    ax.set_xlim(lo, lr["val_loss"].max() + 0.08)
    ax.set_xlabel("validation loss (ниже — лучше)")
    ax.set_title("Learning rate · batch size", loc="left", fontsize=16, color=INK, pad=8)
    for yi, b, v in zip(yy, is_base, lr["val_loss"]):
        if b:
            ax.text(lo + 0.005, yi + 0.45, "исходная настройка", fontsize=12, color=MUTED)
    fig.savefig(out / "tuning.png", dpi=130)
    plt.close(fig)


def choice_chart(res: pd.DataFrame, out: Path) -> None:
    cols = ["chrF++_b1", "ROUGE-L", "BERTScore_F1", "ingredient_mention_%", "health_claim_%"]
    names = ["chrF++ β=1", "ROUGE-L", "BERTScore", "ингредиенты\nв тексте, %", "health claims, %\n(ниже — лучше)"]
    a, b = res.loc[FINAL, cols].to_numpy(float), res.loc[TUNED, cols].to_numpy(float)
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 6.4))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.70, bottom=0.17, wspace=0.35)
    head(fig, "Два кандидата почти равны по метрикам, но расходятся по продукту",
         "Test · одинаковое декодирование, разные веса")
    for ax, m, x1, x2 in zip(axes, names, a, b):
        ax.bar([0, 1], [x1, x2], color=[ORANGE, GREEN], width=0.62)
        for i, v in enumerate([x1, x2]):
            ax.text(i, v * 1.015, f"{v:g}", ha="center", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["v1.0.0 ★", "lr/batch"], fontsize=12.5)
        ax.set_yticks([])
        ax.set_title(m, fontsize=14, color=INK, pad=8)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, max(x1, x2) * 1.18)
    fig.savefig(out / "choice.png", dpi=130)
    plt.close(fig)


def performance_chart(check: dict, live_latency: float | None, out: Path) -> None:
    perf = check["performance"]
    int8 = perf.get("int8", {})
    lab, v, c = [], [], []
    if perf.get("device") == "cuda":
        lab.append("GPU, FP32"); v.append(perf["p50_ms"] / 1000); c.append(LGREY)
        lab.append("CPU, FP32"); v.append(perf["cpu_p50_ms"] / 1000); c.append(GREY)
    else:
        lab.append("CPU, FP32"); v.append(perf["p50_ms"] / 1000); c.append(GREY)
    if "cpu_p50_ms" in int8:
        lab.append("CPU, INT8"); v.append(int8["cpu_p50_ms"] / 1000); c.append(ORANGE)
    if live_latency:
        lab.append("HF Space, INT8\n(живой запрос)"); v.append(live_latency); c.append(ORANGE)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=0.15, right=0.98, top=0.72, bottom=0.14, wspace=0.3)
    head(fig, "INT8 на CPU: быстрее, качество не падает",
         "Один запрос, beams 4 · медиана из inference_check.json · chrF++ на 50 test-примерах")
    ax = axes[0]
    yy = np.arange(len(v))[::-1]
    ax.barh(yy, v, color=c, height=0.62)
    for yi, x in zip(yy, v):
        ax.text(x + 0.04, yi, f"{x:.2f} с", va="center", fontsize=14)
    ax.set_yticks(yy)
    ax.set_yticklabels(lab, fontsize=13.5)
    ax.set_xlim(0, max(v) * 1.18)
    ax.set_xlabel("секунд на описание")
    ax.set_title("Задержка", loc="left", fontsize=16, color=INK, pad=8)
    ax = axes[1]
    if "chrF++_b1_fp32" in int8:
        q = [int8["chrF++_b1_fp32"], int8["chrF++_b1_int8"]]
        ax.bar([0, 1], q, color=[GREY, ORANGE], width=0.6)
        for i, x in enumerate(q):
            ax.text(i, x + 0.3, f"{x:.2f}", ha="center", fontsize=14)
        ax.set_xticks([0, 1])
        # INT8 хранит веса Linear-слоёв в 1 байте вместо 4; ≈310 МБ — размер state_dict после квантования
        ax.set_xticklabels([f"FP32\n{perf['weights_mb']} МБ", "INT8\n≈310 МБ"], fontsize=13.5)
        ax.set_ylim(0, max(q) * 1.2)
        ax.set_ylabel("chrF++ (β=1)")
    ax.set_title("Качество и размер весов", loc="left", fontsize=16, color=INK, pad=8)
    fig.savefig(out / "performance.png", dpi=130)
    plt.close(fig)


def limits_chart(res: pd.DataFrame, bycat: pd.DataFrame, ref_health: float | None, out: Path) -> None:
    col = "chrF ruT5 + tuned decoding"
    bc = bycat.sort_values(col)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.72, bottom=0.14, wspace=0.42)
    head(fig, "Где модель слабее", "Финальная модель на test · слева: по категориям блюд · справа: продуктовые проверки")
    ax = axes[0]
    yy = np.arange(len(bc))
    vals = bc[col].to_numpy(float)
    med = np.median(vals)
    ax.barh(yy, vals, color=[GREY if x < med else BLUE for x in vals], height=0.62)
    for yi, x, k in zip(yy, vals, bc["n"]):
        ax.text(x + 0.2, yi, f"{x}  (n={k})", va="center", fontsize=13)
    ax.set_yticks(yy)
    ax.set_yticklabels(bc["category_top"], fontsize=13.5)
    ax.set_xlim(0, vals.max() * 1.22)
    ax.set_xlabel("chrF++ (β=1)")
    ax.set_title("По категориям", loc="left", fontsize=16, color=INK, pad=8)
    ax = axes[1]
    health = res.loc[FINAL, "health_claim_%"]
    no_ingr = 100 - res.loc[FINAL, "ingredient_mention_%"]
    ax.barh([1, 0], [health, no_ingr], color=ORANGE, height=0.55)
    for yi, x in zip([1, 0], [health, no_ingr]):
        ax.text(x + 0.6, yi, f"{x:.1f}%", va="center", fontsize=14)
    if ref_health is not None:
        ax.plot([ref_health, ref_health], [0.62, 1.38], color=INK, lw=2)
        ax.text(ref_health - 0.5, 0.52, f"в эталонах {ref_health:.1f}%", ha="right", fontsize=12, color=MUTED)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["health claims\n«полезный», «лучший»", "нет ни одного\nингредиента"], fontsize=13.5)
    ax.set_xlim(0, max(health, no_ingr) * 1.45)
    ax.set_xlabel("% ответов на test")
    ax.set_title("Продуктовые проверки", loc="left", fontsize=16, color=INK, pad=8)
    fig.savefig(out / "limits.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--inference", type=Path, default=Path("models/rut5-menu-desc/inference_check.json"))
    ap.add_argument("--out", type=Path, default=Path("charts"))
    ap.add_argument("--live-latency", type=float, default=None, help="задержка живого запроса к Space, секунды")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = args.results

    res = pd.read_csv(R / "final_results.csv").set_index("model")
    diag = pd.read_csv(R / "diagnostics.csv").set_index("model")
    dec = pd.read_csv(R / "tuning_t5_decoding.csv")
    lr = pd.read_csv(R / "tuning_t5_train.csv")
    bycat = pd.read_csv(R / "chrf_by_category.csv")
    cfg_path = R / "config.json"
    base_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    base_cfg = base_cfg.get("base_config", base_cfg)      # config.json или experiments.json-подобная структура
    ref_health = (diag.loc["Reference (test targets)", "health_claim_%"]
                  if "Reference (test targets)" in diag.index else None)

    results_chart(res, args.out)
    beta_chart(diag, args.out)
    waterfall_chart(res, args.out)
    tuning_chart(dec, lr, base_cfg, args.out)
    choice_chart(res, args.out)
    limits_chart(res, bycat, ref_health, args.out)
    if args.inference.exists():
        performance_chart(json.loads(args.inference.read_text(encoding="utf-8")), args.live_latency, args.out)
    else:
        print(f"Нет {args.inference}: performance.png пропущен (выполните Task1_Final_Model.ipynb)")
    print("Графики сохранены в", args.out.resolve())


if __name__ == "__main__":
    main()
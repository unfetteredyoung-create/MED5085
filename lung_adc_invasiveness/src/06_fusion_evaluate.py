"""
Step 6：多模态融合 + 全面评价

参与比较的模型：
  Clinical             临床模型（Step 3）
  Radiomics            影像组学 Rad-score（Step 4）
  DL (image)           纯影像 3D CNN（Step 5）
  DL (image+clinical)  端到端中间层融合网络（Step 5）
  Late fusion          三个单模态模型的概率直接取平均（决策层融合，不需要训练）
  Nomogram             临床独立预测因素 + Rad-score + DL-score 一起做 logistic 回归（列线图）

评价指标：
  AUC（DeLong 95% CI）、ACC / SEN / SPE / PPV / NPV / F1（阈值 = 训练集 Youden 指数，测试集沿用）
  DeLong 检验（Nomogram vs 其他模型）、ROC 曲线、校准曲线 + Brier、决策曲线 DCA、SHAP、列线图
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from utils import MODEL_COLORS, plt  # noqa: E402

Y = config.LABEL_COL
ID = config.ID_COL


def name_of(v):
    return config.DISPLAY_NAMES.get(v, v)


def load_scores():
    meta = utils.impute_clinical(utils.load_meta())
    T = config.TABLE_DIR

    def read(f, cols):
        return pd.read_csv(T / f, dtype={ID: str})[[ID] + cols]

    df = meta.merge(read("scores_clinical.csv", ["clinical_prob"]), on=ID) \
             .merge(read("scores_radiomics.csv", ["rad_score", "radiomics_prob"]), on=ID)
    preds = {"Clinical": "clinical_prob", "Radiomics": "radiomics_prob"}
    for variant, model_name in [("image", "DL (image)"), ("image_clinical", "DL (image+clinical)")]:
        f = T / f"scores_dl_{variant}.csv"
        if f.exists():
            s = read(f.name, ["dl_logit", "dl_prob"]).rename(
                columns={"dl_logit": f"dl_logit_{variant}", "dl_prob": f"dl_prob_{variant}"})
            df = df.merge(s, on=ID)
            preds[model_name] = f"dl_prob_{variant}"
    return df, preds


# ======================= 列线图 =======================
def fit_nomogram(df):
    predictors = json.loads((config.MODEL_DIR / "clinical_model.json").read_text(encoding="utf-8"))["predictors"]
    variables = predictors + ["rad_score"]
    if "dl_logit_image" in df:
        df["dl_score"] = df["dl_logit_image"]
        variables.append("dl_score")
    train = df[df["cohort"] == "train"]
    res = utils.fit_logit(train[variables], train[Y])
    prob = utils.predict_logit(res, df[variables])
    is_train = (df["cohort"] == "train").values
    prob[is_train] = utils.oof_logit(train[variables], train[Y].values)

    ci = np.exp(res.conf_int())
    table = pd.DataFrame({
        "Variable": [name_of(v) for v in variables],
        "Coefficient": res.params[variables].round(4).values,
        "OR (95% CI)": [f"{np.exp(res.params[v]):.2f} ({ci.loc[v, 0]:.2f}-{ci.loc[v, 1]:.2f})" for v in variables],
        "P": [f"{p:.3f}" if p >= 0.001 else "<0.001" for p in res.pvalues[variables]],
    })
    return res, variables, prob, table


def draw_nomogram(res, train, variables):
    beta, b0 = res.params, res.params["const"]
    rows = []
    for v in variables:
        x = train[v]
        binary = set(x.unique()) <= {0, 1}
        lo, hi = (0.0, 1.0) if binary else (x.quantile(0.01), x.quantile(0.99))
        ticks = [0, 1] if binary else [t for t in MaxNLocator(6).tick_values(lo, hi) if lo <= t <= hi]
        rows.append({"var": v, "b": beta[v], "binary": binary, "lo": lo, "hi": hi, "ticks": ticks,
                     "cmin": min(beta[v] * lo, beta[v] * hi), "range": abs(beta[v]) * (hi - lo)})
    max_range = max(r["range"] for r in rows)
    total_max = sum(r["range"] for r in rows) / max_range * 100
    lp0 = b0 + sum(r["cmin"] for r in rows)   # 总分 = 0 时的线性预测值

    def points(r, x):
        return (r["b"] * x - r["cmin"]) / max_range * 100

    n = len(rows) + 3
    fig, ax = plt.subplots(figsize=(9, 0.5 * n + 0.6))

    def axis(y, label, xs, labels):
        ax.text(-4, y, label, ha="right", va="center", fontsize=9, fontweight="bold")
        ax.plot([min(xs), max(xs)], [y, y], color="k", lw=1)
        for x, t in zip(xs, labels):
            ax.plot([x, x], [y, y + 0.12], color="k", lw=1)
            ax.text(x, y + 0.18, t, ha="center", va="bottom", fontsize=7.5)

    axis(0, "Points", list(range(0, 101, 10)), [str(t) for t in range(0, 101, 10)])
    for k, r in enumerate(rows, 1):
        lab = (["No", "Yes"] if r["var"] not in ("sex",) else ["Female", "Male"]) if r["binary"] \
            else [f"{t:g}" for t in r["ticks"]]
        axis(-k, name_of(r["var"]), [points(r, t) for t in r["ticks"]], lab)
    tot_ticks = [t for t in MaxNLocator(8).tick_values(0, total_max) if 0 <= t <= total_max]
    axis(-(len(rows) + 1), "Total points", [t / total_max * 100 for t in tot_ticks], [f"{t:g}" for t in tot_ticks])
    probs, xs = [], []
    for p in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        tp = (utils.logit(p) - lp0) / max_range * 100
        x = tp / total_max * 100
        if 0 <= tp <= total_max and (not xs or x - xs[-1] >= 5):  # 太挤的刻度跳过，避免文字重叠
            probs.append(f"{p:g}")
            xs.append(x)
    if xs:
        axis(-(len(rows) + 2), "Probability of IAC", xs, probs)
    ax.set_xlim(-38, 104)
    ax.set_ylim(-(len(rows) + 2.6), 0.8)
    ax.axis("off")
    ax.set_title("Nomogram for predicting invasive adenocarcinoma (IAC)")
    utils.save_fig(fig, "06_nomogram.png")


def plot_shap(res, df, variables):
    """线性模型的 SHAP 值有解析解：φ_j = β_j × (x_j − 训练集均值)。"""
    import shap
    train = df[df["cohort"] == "train"]
    test = df[df["cohort"] != "train"]
    X = test[variables]
    phi = (X - train[variables].mean()).values * res.params[variables].values
    plt.figure()
    shap.summary_plot(phi, X.values, feature_names=[name_of(v) for v in variables], show=False, plot_size=(7, 4))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP values of the nomogram (test patients)")
    utils.save_fig(fig, "06_shap_nomogram.png")


# ======================= 评价与作图 =======================
def evaluate(df, preds):
    cohorts = [c for c in config.COHORTS if (df["cohort"] == c).any()]
    train = df[df["cohort"] == "train"]
    rows = []
    for model, col in preds.items():
        thr = utils.youden_threshold(train[Y], train[col])
        for c in cohorts:
            s = df[df["cohort"] == c]
            m = utils.binary_metrics(s[Y], s[col], thr)
            rows.append({"Model": model, "Cohort": c, "n": len(s), "Threshold": round(thr, 3),
                         "AUC (95% CI)": f"{m['AUC']:.3f} ({m['AUC_low']:.3f}-{m['AUC_high']:.3f})",
                         **{k: round(v, 3) for k, v in m.items()}})
    perf = pd.DataFrame(rows)

    tests = []
    for c in cohorts:
        s = df[df["cohort"] == c]
        for model, col in preds.items():
            if model != "Nomogram":
                tests.append({"Cohort": c, "Comparison": f"Nomogram vs {model}",
                              "P (DeLong)": round(utils.delong_test(s[Y], s[preds["Nomogram"]], s[col]), 4)})
    return perf, pd.DataFrame(tests), cohorts


def plot_roc(df, preds, cohorts):
    fig, axs = plt.subplots(1, len(cohorts), figsize=(4.6 * len(cohorts), 4.4))
    for ax, c in zip(np.atleast_1d(axs), cohorts):
        s = df[df["cohort"] == c]
        for model, col in preds.items():
            fpr, tpr, _ = roc_curve(s[Y], s[col])
            a = utils.auc(s[Y], s[col])
            ax.plot(fpr, tpr, color=MODEL_COLORS[model], lw=2.2 if model == "Nomogram" else 1.4,
                    label=f"{model}  {a:.3f}")
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=f"ROC - {c} (n={len(s)})")
        ax.legend(loc="lower right", fontsize=7.5, frameon=False, title="AUC", title_fontsize=8)
        ax.set_aspect("equal")
    fig.tight_layout()
    utils.save_fig(fig, "06_roc_curves.png")


def plot_calibration(df, preds, cohorts):
    fig, axs = plt.subplots(1, len(cohorts), figsize=(4.6 * len(cohorts), 4.4))
    for ax, c in zip(np.atleast_1d(axs), cohorts):
        s = df[df["cohort"] == c]
        for model, col in preds.items():
            obs, pred = calibration_curve(s[Y], s[col], n_bins=5, strategy="quantile")
            ax.plot(pred, obs, "o-", ms=3.5, color=MODEL_COLORS[model], lw=2.2 if model == "Nomogram" else 1.2,
                    label=f"{model}  {brier_score_loss(s[Y], s[col]):.3f}")
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        ax.set(xlabel="Predicted probability", ylabel="Observed proportion of IAC", title=f"Calibration - {c}",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=7.5, frameon=False, title="Brier score", title_fontsize=8)
        ax.set_aspect("equal")
    fig.tight_layout()
    utils.save_fig(fig, "06_calibration.png")


def plot_dca(df, preds, cohorts):
    th = np.linspace(0.01, 0.95, 95)
    fig, axs = plt.subplots(1, len(cohorts), figsize=(4.6 * len(cohorts), 4.0))
    for ax, c in zip(np.atleast_1d(axs), cohorts):
        s = df[df["cohort"] == c]
        prev = s[Y].mean()
        for model, col in preds.items():
            ax.plot(th, utils.net_benefit(s[Y], s[col], th), color=MODEL_COLORS[model],
                    lw=2.2 if model == "Nomogram" else 1.2, label=model)
        ax.plot(th, prev - (1 - prev) * th / (1 - th), color="grey", lw=1, label="Treat all")
        ax.axhline(0, color="k", lw=1, label="Treat none")
        ax.set(xlabel="Threshold probability", ylabel="Net benefit", title=f"Decision curve - {c}",
               ylim=(-0.05, prev + 0.05))
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    utils.save_fig(fig, "06_decision_curves.png")


def plot_auc_summary(perf, cohorts):
    models = perf["Model"].unique()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / len(cohorts)
    shades = {"train": 0.35, "internal_test": 0.7, "external_test": 1.0}
    for j, c in enumerate(cohorts):
        p = perf[perf["Cohort"] == c].set_index("Model").loc[models]
        x = np.arange(len(models)) + (j - (len(cohorts) - 1) / 2) * w
        ax.bar(x, p["AUC"], w * 0.9, color=[MODEL_COLORS[m] for m in models], alpha=shades.get(c, 1), label=c)
        ax.errorbar(x, p["AUC"], yerr=[p["AUC"] - p["AUC_low"], p["AUC_high"] - p["AUC"]],
                    fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=15, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (95% CI)")
    ax.set_title("Model comparison (light → dark: train OOF / internal test / external test)")
    fig.tight_layout()
    utils.save_fig(fig, "06_auc_summary.png")


def write_summary(perf, tests, nomo_table, cohorts):
    pivot = perf.pivot(index="Model", columns="Cohort", values="AUC (95% CI)")[cohorts]
    pivot = pivot.loc[perf["Model"].unique()]
    lines = [f"# 结果汇总：实验 {config.EXP_NAME}（自动生成）", "",
             f"相对 config.py 默认值的改动：`{json.dumps(config.EXP_OVERRIDES, ensure_ascii=False) or '{}'}`", ""]
    if utils.is_demo():
        lines += ["> **注意：当前结果基于模拟数据，只用于验证代码流程，不代表任何真实医学结论。**", ""]
    lines += ["## AUC (95% CI)", "", "| Model | " + " | ".join(cohorts) + " |",
              "|---|" + "---|" * len(cohorts)]
    lines += [f"| {m} | " + " | ".join(pivot.loc[m]) + " |" for m in pivot.index]
    lines += ["", "## Nomogram 多因素 logistic 回归", "", "| Variable | OR (95% CI) | P |", "|---|---|---|"]
    lines += [f"| {r.Variable} | {r._3} | {r.P} |" for r in nomo_table.itertuples()]
    lines += ["", "## DeLong 检验", "", "| Cohort | Comparison | P |", "|---|---|---|"]
    lines += [f"| {r.Cohort} | {r.Comparison} | {r._3} |" for r in tests.itertuples()]
    path = config.OUT_DIR / "results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    utils.ensure_dirs()
    df, preds = load_scores()
    res, variables, nomo_prob, nomo_table = fit_nomogram(df)
    df["nomogram_prob"] = nomo_prob
    df["late_fusion_prob"] = df[[preds[m] for m in ["Clinical", "Radiomics", "DL (image)"] if m in preds]].mean(1)
    preds["Late fusion"] = "late_fusion_prob"
    preds["Nomogram"] = "nomogram_prob"
    utils.save_table(nomo_table, "nomogram_logistic_regression.csv")
    utils.save_table(df[[ID, "cohort", Y] + list(preds.values())], "scores_all_models.csv")

    perf, tests, cohorts = evaluate(df, preds)
    utils.save_table(perf, "performance.csv")
    utils.save_table(tests, "delong_tests.csv")

    plot_roc(df, preds, cohorts)
    plot_calibration(df, preds, cohorts)
    plot_dca(df, preds, cohorts)
    plot_auc_summary(perf, cohorts)
    draw_nomogram(res, df[df["cohort"] == "train"], variables)
    plot_shap(res, df, variables)

    print("Nomogram 多因素 logistic 回归：")
    print(nomo_table.to_string(index=False))
    print("\nAUC (95% CI)：")
    print(perf.pivot(index="Model", columns="Cohort", values="AUC (95% CI)")[cohorts]
          .loc[perf["Model"].unique()].to_string())
    print("\nDeLong 检验：")
    print(tests.to_string(index=False))
    print("\n汇总：", write_summary(perf, tests, nomo_table, cohorts))


if __name__ == "__main__":
    main()

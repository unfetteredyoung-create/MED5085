"""
Step 3：临床模型（临床资料 + CT 语义特征）

  1) 基线资料表 Table 1：各队列内 AIS/MIA 与 IAC 两组比较
  2) 训练集单因素 logistic 回归 → p < 0.05 的变量进入多因素 → 向后剔除 → 独立预测因素（OR、95% CI、P）
  3) 用独立预测因素建临床模型，得到每个病人的预测概率
     （训练集用 5 折“折外”预测，测试集用全部训练集拟合的模型）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402

Y = config.LABEL_COL


def fmt_p(p):
    return "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def group_p(sub, var, typ):
    g0, g1 = sub.loc[sub[Y] == 0, var].dropna(), sub.loc[sub[Y] == 1, var].dropna()
    if len(g0) < 2 or len(g1) < 2:
        return np.nan
    if typ == "num":
        return stats.mannwhitneyu(g0, g1).pvalue
    tab = pd.crosstab(sub[var], sub[Y])
    if tab.shape != (2, 2):
        return np.nan
    if (stats.contingency.expected_freq(tab) < 5).any():
        return stats.fisher_exact(tab)[1]
    return stats.chi2_contingency(tab)[1]


def table1(meta):
    def describe(s, typ):
        s = s.dropna()
        if typ == "num":
            return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"
        return f"{int(s.sum())} ({100 * s.mean():.1f}%)"

    rows = []
    for var, typ in config.CLINICAL_VARS.items():
        row = {"Variable": config.DISPLAY_NAMES.get(var, var) + (", median (IQR)" if typ == "num" else ", n (%)")}
        for c in config.COHORTS:
            sub = meta[meta["cohort"] == c]
            if sub.empty:
                continue
            for lab, name in [(0, "AIS/MIA"), (1, "IAC")]:
                g = sub[sub[Y] == lab]
                row[f"{c} {name} (n={len(g)})"] = describe(g[var], typ)
            row[f"{c} P"] = fmt_p(group_p(sub, var, typ))
        rows.append(row)
    return pd.DataFrame(rows)


def select_predictors(train):
    """单因素 → 多因素向后剔除。返回 (结果表, 入选变量列表)。"""
    y = train[Y].values
    uni = {}
    for var in config.CLINICAL_VARS:
        res = utils.fit_logit(train[[var]], y)
        ci = np.exp(res.conf_int().loc[var])
        uni[var] = (np.exp(res.params[var]), ci[0], ci[1], res.pvalues[var])

    cands = [v for v, r in uni.items() if r[3] < config.UNIVARIATE_P]
    if not cands:  # 一个都不显著时，至少保留最显著的一个
        cands = [min(uni, key=lambda v: uni[v][3])]
    while True:
        res = utils.fit_logit(train[cands], y)
        pv = res.pvalues.drop("const")
        if pv.max() <= config.MULTIVARIATE_P or len(cands) == 1:
            break
        cands.remove(pv.idxmax())

    rows = []
    for var in config.CLINICAL_VARS:
        o, lo, hi, p = uni[var]
        row = {"Variable": config.DISPLAY_NAMES.get(var, var),
               "Univariate OR (95% CI)": f"{o:.2f} ({lo:.2f}-{hi:.2f})", "Univariate P": fmt_p(p)}
        if var in cands:
            ci = np.exp(res.conf_int().loc[var])
            row["Multivariable OR (95% CI)"] = f"{np.exp(res.params[var]):.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
            row["Multivariable P"] = fmt_p(res.pvalues[var])
        rows.append(row)
    return pd.DataFrame(rows).fillna(""), cands


def main():
    utils.ensure_dirs()
    raw_meta = utils.load_meta()
    t1 = table1(raw_meta)  # Table 1 用填补前的原始数据
    utils.save_table(t1, "table1_baseline.csv")
    print("Table 1 已保存（各队列 AIS/MIA vs IAC 比较）")

    meta = utils.impute_clinical(raw_meta)
    train = meta[meta["cohort"] == "train"]
    lr_table, predictors = select_predictors(train)
    utils.save_table(lr_table, "clinical_logistic_regression.csv")
    print(lr_table.to_string(index=False))
    print(f"\n独立预测因素：{predictors}")

    final = utils.fit_logit(train[predictors], train[Y])
    prob = utils.predict_logit(final, meta[predictors])
    is_train = (meta["cohort"] == "train").values
    prob[is_train] = utils.oof_logit(train[predictors], train[Y].values)

    scores = meta[[config.ID_COL, "cohort", Y]].assign(clinical_prob=prob)
    utils.save_table(scores, "scores_clinical.csv")
    (config.MODEL_DIR / "clinical_model.json").write_text(json.dumps({
        "predictors": predictors,
        "coef": final.params.to_dict(),
    }, indent=2), encoding="utf-8")

    for c in config.COHORTS:
        s = scores[scores["cohort"] == c]
        if len(s):
            print(f"  Clinical AUC [{c}]: {utils.auc(s[Y], s['clinical_prob']):.3f}")


if __name__ == "__main__":
    main()

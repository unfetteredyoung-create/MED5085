"""
Step 4：影像组学标签（Rad-score）

特征筛选“漏斗”（全部只在训练集上做，测试集只拿来算分）：
  ① ICC >= 0.75 的稳定特征，且无缺失、非常数
  ② Z-score 标准化
  ③ Mann-Whitney U 检验 p < 0.05
  ④ 去冗余：两两 |Pearson r| > 0.9 时只保留 p 值更小的
  ⑤ LASSO logistic 回归（5 折交叉验证选 λ，1-SE 规则 → 更稀疏）
Rad-score = 截距 + Σ 系数 × 标准化特征值（就是 LASSO 模型的线性预测值）
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from utils import plt  # noqa: E402

Y = config.LABEL_COL


def skf():
    return StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)


class RadiomicsSelector(BaseEstimator, TransformerMixin):
    """上面的 ①~④ 步。输入/输出都是 DataFrame。"""

    def __init__(self, stable=None, p_thr=config.MWU_P, corr_thr=config.CORR_THRESHOLD, min_keep=10):
        self.stable = stable
        self.p_thr = p_thr
        self.corr_thr = corr_thr
        self.min_keep = min_keep

    def fit(self, X, y):
        y = np.asarray(y)
        funnel = {"all": X.shape[1]}
        if self.stable is not None:
            X = X[[c for c in X.columns if c in set(self.stable)]]
        X = X.loc[:, np.isfinite(X).all() & (X.std() > 1e-8)]
        funnel["ICC stable"] = X.shape[1]

        self.mean_, self.std_ = X.mean(), X.std()
        Z = (X - self.mean_) / self.std_
        p = pd.Series(mannwhitneyu(Z[y == 0], Z[y == 1], axis=0).pvalue, index=Z.columns).sort_values()
        cand = p[p < self.p_thr].index.tolist()
        if len(cand) < self.min_keep:
            cand = p.index[:self.min_keep].tolist()
        funnel["Mann-Whitney U"] = len(cand)

        corr = Z[cand].corr().abs().values
        keep = []
        for i in range(len(cand)):  # cand 已按 p 值从小到大排好
            if all(corr[i, j] <= self.corr_thr for j in keep):
                keep.append(i)
        self.selected_ = [cand[i] for i in keep]
        funnel["De-correlation"] = len(self.selected_)
        self.funnel_ = funnel
        return self

    def transform(self, X):
        s = self.selected_
        return (X[s] - self.mean_[s]) / self.std_[s]


def l1_logistic(C):
    # liblinear 默认会把截距也一起惩罚；intercept_scaling 调大后截距几乎不受惩罚
    return LogisticRegression(penalty="l1", C=C, solver="liblinear", intercept_scaling=100, max_iter=5000)


class LassoLR(BaseEstimator, ClassifierMixin):
    """L1 正则 logistic 回归；用 5 折交叉验证的 deviance 选 λ（1-SE 规则）。"""

    def __init__(self, n_lambdas=40, one_se=True):
        self.n_lambdas = n_lambdas
        self.one_se = one_se

    def fit(self, X, y):
        Cs = np.logspace(-2, 1.5, self.n_lambdas)   # C = 1/λ
        cv = LogisticRegressionCV(Cs=Cs, cv=skf(), penalty="l1", solver="liblinear",
                                  scoring="neg_log_loss", intercept_scaling=100, max_iter=5000).fit(X, y)
        dev = -2 * cv.scores_[1]                    # folds x Cs，binomial deviance
        mean, se = dev.mean(0), dev.std(0) / np.sqrt(dev.shape[0])
        best = int(np.argmin(mean))
        idx = best
        if self.one_se:  # 在“不比最优差一个标准误”的模型里选惩罚最强（最稀疏）的
            idx = int(np.where(mean <= mean[best] + se[best])[0].min())
        model = l1_logistic(Cs[idx]).fit(X, y)
        if not np.any(model.coef_):  # 惩罚太强、一个特征都没选上时退回最优 λ
            idx = best
            model = l1_logistic(Cs[idx]).fit(X, y)
        self.Cs_, self.cv_mean_, self.cv_se_, self.best_idx_, self.idx_ = Cs, mean, se, best, idx
        self.model_ = model
        self.classes_ = model.classes_
        return self

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def plot_lasso(pipe, Xtr, ytr):
    sel, lasso = pipe.named_steps["select"], pipe.named_steps["lasso"]
    Z = sel.transform(Xtr)
    log_lam = np.log10(1 / lasso.Cs_)
    paths = np.array([l1_logistic(C).fit(Z, ytr).coef_[0] for C in lasso.Cs_])

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axs[0]
    ax.errorbar(log_lam, lasso.cv_mean_, yerr=lasso.cv_se_, fmt="o", ms=3, color="#C44E52",
                ecolor="#bbbbbb", capsize=2)
    for i, ls in [(lasso.best_idx_, ":"), (lasso.idx_, "--")]:
        ax.axvline(log_lam[i], color="k", ls=ls, lw=1)
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("Binomial deviance (5-fold CV)")
    ax.set_title("LASSO tuning (dotted: min, dashed: 1-SE)")
    ax = axs[1]
    ax.plot(log_lam, paths, lw=1)
    ax.axvline(log_lam[lasso.idx_], color="k", ls="--", lw=1)
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("Coefficient")
    ax.set_title("LASSO coefficient paths")
    fig.tight_layout()
    utils.save_fig(fig, "04_lasso.png")


def plot_signature(sig):
    s = sig[sig["feature"] != "(Intercept)"].sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
    ax.barh(s["feature"], s["coefficient"], color=np.where(s["coefficient"] > 0, "#C44E52", "#4C72B0"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("LASSO coefficient")
    ax.set_title(f"Radiomics signature ({len(s)} features)")
    ax.tick_params(axis="y", labelsize=7)
    utils.save_fig(fig, "04_rad_signature.png")


def plot_waterfall(scores):
    cohorts = [c for c in config.COHORTS if (scores["cohort"] == c).any()]
    fig, axs = plt.subplots(1, len(cohorts), figsize=(4.2 * len(cohorts), 3.2), sharey=True)
    for ax, c in zip(np.atleast_1d(axs), cohorts):
        s = scores[scores["cohort"] == c].sort_values("rad_score").reset_index(drop=True)
        ax.bar(s.index, s["rad_score"], width=1.0,
               color=np.where(s[Y] == 1, "#C44E52", "#4C72B0"))
        ax.set_title(f"{c} (red = IAC, blue = AIS/MIA)", fontsize=9)
        ax.set_xticks([])
    np.atleast_1d(axs)[0].set_ylabel("Rad-score")
    fig.tight_layout()
    utils.save_fig(fig, "04_rad_score_waterfall.png")


def main():
    utils.ensure_dirs()
    meta = utils.load_meta()
    feats = pd.read_csv(config.TABLE_DIR / "radiomics_features.csv", dtype={config.ID_COL: str}) \
        .set_index(config.ID_COL).loc[meta[config.ID_COL]]
    icc = pd.read_csv(config.TABLE_DIR / "radiomics_icc.csv")
    stable = icc.loc[icc["icc"] >= config.ICC_THRESHOLD, "feature"].tolist()

    is_train = (meta["cohort"] == "train").values
    Xtr, ytr = feats[is_train], meta.loc[is_train, Y].values

    pipe = Pipeline([("select", RadiomicsSelector(stable=stable)), ("lasso", LassoLR(one_se=config.LASSO_ONE_SE))])
    pipe.fit(Xtr, ytr)
    funnel = {**pipe.named_steps["select"].funnel_,
              "LASSO": int(np.count_nonzero(pipe.named_steps["lasso"].model_.coef_))}
    print("特征筛选漏斗：", " -> ".join(f"{k}: {v}" for k, v in funnel.items()))

    lasso = pipe.named_steps["lasso"].model_
    names = pipe.named_steps["select"].selected_
    sig = pd.DataFrame({"feature": names, "coefficient": lasso.coef_[0]})
    sig = sig[sig["coefficient"] != 0]
    sig = pd.concat([pd.DataFrame({"feature": ["(Intercept)"], "coefficient": lasso.intercept_}), sig])
    utils.save_table(sig, "radiomics_signature.csv")
    print("Rad-score =", " ".join(f"{c:+.3f}*[{f}]" if f != "(Intercept)" else f"{c:.3f}"
                                  for f, c in zip(sig["feature"], sig["coefficient"])))

    rad = pipe.decision_function(feats)
    rad[is_train] = cross_val_predict(pipe, Xtr, ytr, cv=skf(), method="decision_function")
    scores = meta[[config.ID_COL, "cohort", Y]].assign(rad_score=rad, radiomics_prob=utils.sigmoid(rad))
    utils.save_table(scores, "scores_radiomics.csv")
    joblib.dump(pipe, config.MODEL_DIR / "radiomics_pipeline.joblib")

    plot_lasso(pipe, Xtr, ytr)
    plot_signature(sig)
    plot_waterfall(scores)
    for c in config.COHORTS:
        s = scores[scores["cohort"] == c]
        if len(s):
            print(f"  Radiomics AUC [{c}]: {utils.auc(s[Y], s['rad_score']):.3f}")


if __name__ == "__main__":
    main()

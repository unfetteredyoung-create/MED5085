"""
各步骤共用的小工具：读写 NIfTI、重采样/裁剪、评价指标（DeLong、Youden、DCA）、画图保存。
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 不弹窗，直接存图（服务器上也能跑）
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage, stats  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402


# ======================= 通用 =======================
def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs():
    for d in [config.PROCESSED_DIR, config.PATCH_DIR, config.FIG_DIR, config.TABLE_DIR, config.MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def is_demo():
    return config.DEMO_FLAG.exists()


def load_meta():
    return pd.read_csv(config.META_CSV, dtype={config.ID_COL: str})


def save_table(df, name, index=False):
    path = config.TABLE_DIR / name
    df.to_csv(path, index=index, encoding="utf-8-sig")  # utf-8-sig：Excel 直接打开不乱码
    return path


def sigmoid(x):
    return expit(x)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def impute_clinical(meta):
    """临床缺失值填补：连续变量用训练集中位数、二分类变量用训练集众数（只看训练集，避免信息泄露）。"""
    meta = meta.copy()
    train = meta[meta["cohort"] == "train"]
    for var, typ in config.CLINICAL_VARS.items():
        fill = train[var].median() if typ == "num" else train[var].mode().iloc[0]
        meta[var] = meta[var].fillna(fill)
    return meta


# ======================= Logistic 回归 =======================
def fit_logit(X, y):
    import statsmodels.api as sm
    return sm.Logit(np.asarray(y, float), sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)


def predict_logit(res, X):
    import statsmodels.api as sm
    return np.asarray(res.predict(sm.add_constant(X, has_constant="add")))


def oof_logit(X, y, folds=config.CV_FOLDS):
    """训练集 K 折交叉验证的“折外”预测：每个病人的概率都来自没见过他的模型。"""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    out = np.zeros(len(y))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=config.SEED).split(X, y):
        out[va] = predict_logit(fit_logit(X.iloc[tr], y[tr]), X.iloc[va])
    return out


# ======================= 影像 IO 与预处理 =======================
def load_nifti(path):
    """返回 (数组 [x, y, z], 体素间距 mm)。用 nibabel 读，中文路径也没问题。"""
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return arr, spacing


def save_nifti(arr, spacing, path):
    import nibabel as nib
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def resample(arr, spacing, target_spacing, order=1):
    """把体数据从原始 spacing 重采样到 target_spacing。图像用线性插值(order=1)。"""
    factors = np.array(spacing) / np.array(target_spacing)
    return ndimage.zoom(arr, factors, order=order)


def crop_center(arr, center, size, pad_value):
    """以 center 为中心裁出 size^3 的立方块，超出边界的部分用 pad_value 填充。"""
    out = np.full((size,) * 3, pad_value, dtype=arr.dtype)
    start = np.array(center) - size // 2
    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + size, arr.shape)
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        arr[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def window_normalize(img_hu, window=config.HU_WINDOW):
    lo, hi = window
    return (np.clip(img_hu, lo, hi) - lo) / (hi - lo)


# ======================= 评价指标 =======================
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n)
    out[order] = t + 1
    return out


def _fast_delong(preds, m):
    """Sun & Xu (2014) 快速 DeLong。preds: k x n，前 m 列为阳性样本。"""
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def _sort_pos_first(y):
    y = np.asarray(y).astype(int)
    return np.argsort(-y, kind="mergesort"), int(y.sum())


def delong_auc_ci(y, p, alpha=0.95):
    """AUC 及 DeLong 法 95% 置信区间。"""
    order, m = _sort_pos_first(y)
    aucs, cov = _fast_delong(np.asarray(p, float)[order][None, :], m)
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    return aucs[0], max(0.0, aucs[0] - z * se), min(1.0, aucs[0] + z * se)


def delong_test(y, p1, p2):
    """比较两个模型在同一批病人上的 AUC，返回双侧 p 值。"""
    order, m = _sort_pos_first(y)
    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 1.0
    z = abs(aucs[0] - aucs[1]) / np.sqrt(var)
    return float(2 * stats.norm.sf(z))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def binary_metrics(y, p, threshold):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    auc, lo, hi = delong_auc_ci(y, p)
    return {
        "AUC": auc, "AUC_low": lo, "AUC_high": hi,
        "ACC": div(tp + tn, len(y)), "SEN": div(tp, tp + fn), "SPE": div(tn, tn + fp),
        "PPV": div(tp, tp + fp), "NPV": div(tn, tn + fn), "F1": div(2 * tp, 2 * tp + fp + fn),
    }


def net_benefit(y, p, thresholds):
    """决策曲线分析 (DCA) 的净获益。"""
    y = np.asarray(y).astype(int)
    n = len(y)
    out = []
    for pt in thresholds:
        pred = np.asarray(p) >= pt
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        out.append(tp / n - fp / n * pt / (1 - pt))
    return np.array(out)


def auc(y, p):
    return roc_auc_score(y, p)


# ======================= 画图 =======================
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# 模型配色固定下来，所有图保持一致
MODEL_COLORS = {
    "Clinical": "#4C72B0",
    "Radiomics": "#55A868",
    "DL (image)": "#C44E52",
    "DL (image+clinical)": "#8172B3",
    "Late fusion": "#937860",
    "Nomogram": "#DD8452",
}


def save_fig(fig, name, folder=None):
    """存图；如果当前是模拟数据，加水印，防止被误当成真实结果。"""
    if is_demo():
        fig.text(0.5, 0.5, "SIMULATED DATA", fontsize=46, color="grey", alpha=0.10,
                 ha="center", va="center", rotation=25, zorder=100)
        fig.text(0.995, 0.005, "Demo on simulated data - not real results", fontsize=7,
                 color="grey", ha="right", va="bottom")
    path = (folder or config.FIG_DIR) / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

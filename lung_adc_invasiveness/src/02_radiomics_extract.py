"""
Step 2：用 PyRadiomics 提取影像组学特征 + 勾画稳定性（ICC）分析

- 特征：形状 / 一阶统计 / 纹理（GLCM、GLRLM、GLSZM、GLDM、NGTDM），
  在原图、LoG 滤波图、小波分解图上分别计算，约 1000+ 个特征。
- ICC：同一批病例由两位医生分别勾画，各提一遍特征，ICC >= 0.75 的特征才算“稳定”。
  如果 data/raw/masks_reader2/ 不存在，就把 mask 随机外扩/内缩一个体素来模拟第二位医生。
"""
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from utils import plt  # noqa: E402

_extractor = None  # 每个子进程各自建一个提取器


def get_extractor():
    global _extractor
    if _extractor is None:
        import radiomics
        from radiomics import featureextractor
        radiomics.setVerbosity(logging.ERROR)
        ex = featureextractor.RadiomicsFeatureExtractor(**config.RADIOMICS_SETTINGS)
        ex.disableAllImageTypes()
        for name, kwargs in config.RADIOMICS_IMAGE_TYPES.items():
            ex.enableImageTypeByName(name, customArgs=kwargs)
        ex.enableAllFeatures()
        _extractor = ex
    return _extractor


def to_sitk(arr, dtype):
    import SimpleITK as sitk
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(dtype))  # sitk 数组顺序是 z,y,x
    im.SetSpacing(tuple(float(s) for s in config.TARGET_SPACING))
    return im


def ball(radius):
    r = int(radius)
    return np.linalg.norm(np.indices((2 * r + 1,) * 3) - r, axis=0) <= radius


def run_extractor(image, mask):
    res = get_extractor().execute(image, to_sitk(mask, np.uint8), label=1)
    return {k: float(v) for k, v in res.items() if not k.startswith("diagnostics_")}


def extract_one(args):
    pid, mask_key, perturb_seed = args
    d = np.load(config.PATCH_DIR / f"{pid}.npz")
    mask = d[mask_key].astype(bool)
    if perturb_seed is not None:
        rng = np.random.default_rng(perturb_seed)
        eroded = ndimage.binary_erosion(mask)
        mask = ndimage.binary_dilation(mask) if (rng.random() < 0.5 or eroded.sum() < 20) else eroded
    image = to_sitk(d["image"], np.float32)
    feats = run_extractor(image, mask)
    if config.PERITUMORAL_MM > 0:
        # 瘤周：结节向外扩 r mm 的环形区域（不含结节本身）。形状特征对环没有意义，去掉。
        # 真实数据中贴胸膜的结节，环会伸进胸壁，可再和肺野 mask 取交集。
        ring = ndimage.binary_dilation(mask, structure=ball(config.PERITUMORAL_MM / config.TARGET_SPACING[0])) & ~mask
        feats.update({f"peri_{k}": v for k, v in run_extractor(image, ring).items() if "_shape_" not in k})
    return pid, feats


def run_parallel(jobs):
    with ProcessPoolExecutor(config.N_WORKERS) as ex:
        out = dict(ex.map(extract_one, jobs, chunksize=2))
    return pd.DataFrame.from_dict(out, orient="index")


def icc_2_1(a, b):
    """ICC(2,1)：双向随机效应、绝对一致性、单次测量。a, b 形状 (病例数, 特征数)。"""
    Y = np.stack([a, b], axis=1)
    n, k = Y.shape[:2]
    grand = Y.mean(axis=(0, 1))
    row_m = Y.mean(axis=1)
    col_m = Y.mean(axis=0)
    msr = k * ((row_m - grand) ** 2).sum(0) / (n - 1)
    msc = n * ((col_m - grand) ** 2).sum(0) / (k - 1)
    mse = ((Y - row_m[:, None] - col_m[None] + grand) ** 2).sum((0, 1)) / ((n - 1) * (k - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def main():
    utils.ensure_dirs()
    meta = utils.load_meta()
    pids = meta[config.ID_COL].tolist()

    t0 = time.time()
    print(f"提取 {len(pids)} 个病例的影像组学特征（图像类型：{list(config.RADIOMICS_IMAGE_TYPES)}，"
          f"瘤周：{f'{config.PERITUMORAL_MM} mm' if config.PERITUMORAL_MM else '无'}）...")
    feats = run_parallel([(p, "mask", None) for p in pids]).loc[pids]
    feats.index.name = config.ID_COL
    utils.save_table(feats, "radiomics_features.csv", index=True)
    print(f"  得到 {feats.shape[1]} 个特征，用时 {time.time() - t0:.0f}s")

    # ---------- ICC 稳定性分析（只用训练集病例） ----------
    train_pids = meta.loc[meta["cohort"] == "train", config.ID_COL].values
    rng = np.random.default_rng(config.SEED)
    icc_pids = list(rng.choice(train_pids, size=min(config.ICC_N_CASES, len(train_pids)), replace=False))
    has_reader2 = all("mask2" in np.load(config.PATCH_DIR / f"{p}.npz").files for p in icc_pids)
    if has_reader2:
        print(f"用第二位医生的勾画计算 ICC（{len(icc_pids)} 例）...")
        jobs = [(p, "mask2", None) for p in icc_pids]
    else:
        print(f"没有第二位医生的勾画，随机扰动 mask 模拟（{len(icc_pids)} 例）...")
        jobs = [(p, "mask", 1000 + i) for i, p in enumerate(icc_pids)]
    feats2 = run_parallel(jobs).loc[icc_pids, feats.columns]
    icc = pd.DataFrame({"feature": feats.columns,
                        "icc": icc_2_1(feats.loc[icc_pids].values, feats2.values)})
    utils.save_table(icc, "radiomics_icc.csv")
    n_stable = int((icc["icc"] >= config.ICC_THRESHOLD).sum())
    print(f"  ICC >= {config.ICC_THRESHOLD} 的稳定特征：{n_stable}/{len(icc)}")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(icc["icc"].clip(-0.2, 1).dropna(), bins=40, color="#4C72B0")
    ax.axvline(config.ICC_THRESHOLD, color="#C44E52", ls="--", label=f"ICC = {config.ICC_THRESHOLD}")
    ax.set_xlabel("ICC(2,1) between two segmentations")
    ax.set_ylabel("Number of features")
    ax.set_title(f"Feature stability: {n_stable}/{len(icc)} features kept")
    ax.legend(frameon=False)
    utils.save_fig(fig, "02_radiomics_icc.png")
    print(f"总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

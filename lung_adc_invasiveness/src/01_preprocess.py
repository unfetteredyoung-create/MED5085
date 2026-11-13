"""
Step 1：影像预处理 + 队列划分

对每个病人：
  1. 读入 CT 和结节 mask（NIfTI）
  2. 重采样到 1x1x1 mm（不同医院/层厚的 CT 统一尺度）
  3. 以结节为中心裁出 72^3 mm 的立方块，存成 .npz（后面影像组学和深度学习都用它）
然后：
  4. 划分队列：EXTERNAL_CENTERS 里的中心 → 外部验证集；其余按 7:3 分层随机 → 训练集 / 内部测试集
  5. 画一张质控图（QC），肉眼检查勾画是否对齐
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402
from utils import plt  # noqa: E402


def process_case(pid):
    img, sp = utils.load_nifti(config.IMAGE_DIR / f"{pid}.nii.gz")
    msk, sp_m = utils.load_nifti(config.MASK_DIR / f"{pid}.nii.gz")
    if img.shape != msk.shape:
        raise ValueError(f"{pid}: CT 与 mask 尺寸不一致 {img.shape} vs {msk.shape}")

    img_r = utils.resample(img, sp, config.TARGET_SPACING, order=1)
    msk_r = utils.resample((msk > 0.5).astype(np.float32), sp, config.TARGET_SPACING, order=1) > 0.5
    if msk_r.sum() == 0:
        raise ValueError(f"{pid}: mask 为空")

    coords = np.argwhere(msk_r)
    center = (coords.min(0) + coords.max(0)) // 2   # 结节外接框中心
    size = config.CROP_SIZE
    out = {
        "image": utils.crop_center(img_r, center, size, pad_value=-1024).astype(np.int16),
        "mask": utils.crop_center(msk_r.astype(np.uint8), center, size, pad_value=0),
    }
    m2_path = config.MASK2_DIR / f"{pid}.nii.gz"
    if m2_path.exists():  # 第二位医生的勾画（做 ICC 用），同样处理
        m2, _ = utils.load_nifti(m2_path)
        m2_r = utils.resample((m2 > 0.5).astype(np.float32), sp, config.TARGET_SPACING, order=1) > 0.5
        out["mask2"] = utils.crop_center(m2_r.astype(np.uint8), center, size, pad_value=0)
    np.savez_compressed(config.PATCH_DIR / f"{pid}.npz", **out)

    extent = coords.max(0) - coords.min(0) + 1
    return {
        config.ID_COL: pid,
        "orig_spacing_z": round(sp[2], 3),
        "nodule_volume_mm3": float(msk_r.sum() * np.prod(config.TARGET_SPACING)),
        "nodule_extent_max_mm": float(extent.max()),
        "touches_border": bool(extent.max() >= size - 2),   # 结节比裁剪块还大 → 需要调大 CROP_SIZE
    }


def assign_cohorts(df):
    df = df.copy()
    is_ext = df[config.CENTER_COL].isin(config.EXTERNAL_CENTERS) if config.CENTER_COL in df else False
    df["cohort"] = np.where(is_ext, "external_test", "")
    internal = df[df["cohort"] == ""]
    tr_idx, te_idx = train_test_split(
        internal.index, test_size=config.INTERNAL_TEST_SIZE,
        stratify=internal[config.LABEL_COL], random_state=config.SEED)
    df.loc[tr_idx, "cohort"] = "train"
    df.loc[te_idx, "cohort"] = "internal_test"
    return df


def qc_figure(meta, n=24):
    rng = np.random.default_rng(config.SEED)
    pids = rng.choice(meta[config.ID_COL].values, size=min(n, len(meta)), replace=False)
    ncol = 8
    nrow = int(np.ceil(len(pids) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.8, nrow * 2.0))
    for ax, pid in zip(axs.ravel(), pids):
        d = np.load(config.PATCH_DIR / f"{pid}.npz")
        z = int(np.argwhere(d["mask"])[:, 2].mean())
        ax.imshow(utils.window_normalize(d["image"][:, :, z]).T, cmap="gray", vmin=0, vmax=1)
        ax.contour(d["mask"][:, :, z].T, levels=[0.5], colors="#ff3b30", linewidths=0.7)
        row = meta.loc[meta[config.ID_COL] == pid].iloc[0]
        ax.set_title(f"{pid}  {'IAC' if row[config.LABEL_COL] == 1 else 'AIS/MIA'}", fontsize=8)
    for ax in axs.ravel():
        ax.axis("off")
    fig.suptitle("QC: axial slice through nodule center (red = ROI)", y=1.0)
    fig.tight_layout()
    return utils.save_fig(fig, "01_qc_patches.png")


def main():
    utils.ensure_dirs()
    clin = pd.read_csv(config.CLINICAL_CSV, dtype={config.ID_COL: str})
    missing = [p for p in clin[config.ID_COL]
               if not (config.IMAGE_DIR / f"{p}.nii.gz").exists() or not (config.MASK_DIR / f"{p}.nii.gz").exists()]
    if missing:
        print(f"[警告] {len(missing)} 个病人缺 CT 或 mask，已跳过：{missing[:10]} ...")
        clin = clin[~clin[config.ID_COL].isin(missing)]

    pids = clin[config.ID_COL].tolist()
    print(f"预处理 {len(pids)} 个病例：重采样到 {config.TARGET_SPACING} mm，裁剪 {config.CROP_SIZE}^3 ...")
    with ProcessPoolExecutor(config.N_WORKERS) as ex:
        info = pd.DataFrame(list(ex.map(process_case, pids, chunksize=4)))

    meta = assign_cohorts(clin.merge(info, on=config.ID_COL))
    meta.to_csv(config.META_CSV, index=False, encoding="utf-8-sig")

    n_big = int(meta["touches_border"].sum())
    if n_big:
        print(f"[警告] {n_big} 个结节超出 {config.CROP_SIZE} mm 裁剪块，建议调大 config.CROP_SIZE")
    print(meta.groupby("cohort")[config.LABEL_COL].agg(n="count", IAC="sum", IAC_ratio="mean").round(3))
    print("QC 图：", qc_figure(meta))
    print("meta 表：", config.META_CSV)


if __name__ == "__main__":
    main()

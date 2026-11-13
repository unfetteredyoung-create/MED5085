"""
Step 0：生成【模拟】数据，让整条流程在拿到真实数据之前就能跑通、调试。

模拟了两家医院（中心 A：薄层 CT；中心 B：层厚更厚、噪声更大，用作外部验证），每个病人包含：
  - 一个三维 CT（NIfTI，单位 HU）：肺实质背景 + 随机血管 + 一个结节
    （磨玻璃成分 + 实性成分，带分叶、毛刺、空泡，浸润性越强的病人结节越大、实性越多、越不规则）
  - 一个结节勾画 mask（模拟医生手工勾画的误差）
  - 一行临床信息：年龄、性别、吸烟、CEA、CT 语义特征（最大径、CTR、分叶、毛刺…）和病理标签

注意：这里的“规律”是我们人为设定的，跑出来的 AUC 没有任何医学意义，只用来检验代码。
拿到真实数据后，按 README 的格式放进 data/raw/ 即可，不要再运行这个脚本。
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402

CENTERS = {
    # n: 病人数；spacing: 原始体素间距 (x, y, z) mm；noise: 图像噪声 HU；shift: 病例构成差异
    "A": {"n": 260, "spacing": (0.70, 0.70, 1.25), "noise": 18.0, "shift": 0.0},
    "B": {"n": 100, "spacing": (0.78, 0.78, 2.00), "noise": 30.0, "shift": 0.2},
}
FOV_MM = 72.0          # 每个模拟 CT 的视野（立方体边长 mm），真实 CT 是整个胸部
MISSING_CEA = 0.08     # 8% 的病人没测 CEA，模拟真实数据中的缺失


def smooth_noise(shape, sigma_vox, rng):
    """均值 0、标准差 1 的平滑随机场，用来做纹理和不规则边界。"""
    f = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma_vox)
    return (f - f.mean()) / (f.std() + 1e-8)


def soft_tube(P, p0, d, radius, length=None):
    """点集 P 到直线（或线段）的软掩膜，用来画血管和毛刺。"""
    v = P - p0
    t = v @ d
    if length is not None:
        t = np.clip(t, 0, length)
    dist = np.linalg.norm(v - t[..., None] * d, axis=-1)
    return utils.sigmoid((radius - dist) / 0.35)


def unit(rng):
    d = rng.standard_normal(3)
    return d / np.linalg.norm(d)


def sample_patient(rng, shift):
    """先抽一个“潜在浸润程度” z，病理标签和所有影像/临床表现都由 z 加噪声决定。"""
    z = rng.standard_normal() + shift
    y = int(z + rng.normal(0, 0.55) > -0.1)
    n = rng.standard_normal(5)

    p = {}  # 影像真实参数（画图用）
    p["diam_true"] = float(np.clip(12 + 4.0 * (0.65 * z + 0.76 * n[0]), 5, 30))
    ctr = float(utils.sigmoid(-0.6 + 1.8 * (0.75 * z + 0.66 * n[1])))
    p["ctr_true"] = 0.0 if ctr < 0.15 else (1.0 if ctr > 0.92 else ctr)
    p["lob_amp"] = float(0.06 + 0.16 * utils.sigmoid(1.6 * (0.6 * z + 0.8 * n[2])))
    p["tex_sigma"] = float(20 + 60 * utils.sigmoid(1.5 * (0.75 * z + 0.66 * n[3])))  # 只有影像能看到的异质性
    p["n_spikes"] = int(rng.poisson(4 * utils.sigmoid(1.4 * z - 1.2)))
    p["vacuole_true"] = bool(rng.random() < 0.08 + 0.2 * utils.sigmoid(z))

    def misread(v, q):  # 医生阅片有一定误判率
        return int(v) if rng.random() > q else 1 - int(v)

    smoking = int(rng.random() < utils.sigmoid(-1.0 + 0.5 * z))
    # 医生在 2D 层面上目测 CTR，误差不小
    ctr_read = 0.0 if p["ctr_true"] == 0 else float(np.clip(p["ctr_true"] + rng.normal(0, 0.10), 0.05, 1.0))
    clin = {
        "age": int(np.clip(round(57 + 4 * z + rng.normal(0, 9)), 28, 85)),
        "sex": int(rng.random() < 0.25 + 0.45 * smoking),
        "smoking": smoking,
        "cea": round(float(np.exp(0.7 + 0.30 * z + rng.normal(0, 0.45))), 2),
        "diameter_mm": round(p["diam_true"] * rng.normal(1, 0.08), 1),
        "ctr": int(round(100 * ctr_read)),   # 百分比 0-100
        "nodule_type": "pGGN" if ctr_read == 0 else ("solid" if ctr_read >= 0.95 else "mGGN"),
        "lobulation": misread(p["lob_amp"] > 0.14, 0.15),
        "spiculation": misread(p["n_spikes"] >= 2, 0.10),
        "pleural_indentation": int(rng.random() < utils.sigmoid(-1.3 + 0.9 * z)),
        "vacuole": misread(p["vacuole_true"], 0.08),
        "pathology": "IAC" if y else ("AIS" if z < -0.9 else "MIA"),
        "label": y,
    }
    if rng.random() < MISSING_CEA:
        clin["cea"] = np.nan
    return p, clin


def make_volume(p, spacing, noise, seed):
    rng = np.random.default_rng(seed)
    sp = np.array(spacing, dtype=np.float32)
    shape = tuple(int(round(FOV_MM / s)) for s in sp)
    axes = [((np.arange(n) - (n - 1) / 2) * s).astype(np.float32) for n, s in zip(shape, sp)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # 每个体素的物理坐标 (mm)

    # 1) 肺实质背景 + 血管
    vol = -860 + 25 * smooth_noise(shape, 2.0 / sp, rng)
    for _ in range(rng.integers(3, 7)):
        w = soft_tube(P, rng.uniform(-25, 25, 3).astype(np.float32), unit(rng).astype(np.float32),
                      rng.uniform(0.8, 2.0))
        vol = vol + w * (40 - vol)

    # 2) 结节主体（磨玻璃 + 分叶边界）
    c = rng.normal(0, 3, 3).astype(np.float32)
    R = p["diam_true"] / 2
    ax = rng.uniform(0.85, 1.15, 3).astype(np.float32)
    ax /= np.cbrt(ax.prod())
    rho = np.linalg.norm((P - c) / ax, axis=-1)
    lob = smooth_noise(shape, max(R * 0.6, 2.0) / sp, rng)
    body = rho < R * (1 + p["lob_amp"] * lob)
    lab, n_cc = ndimage.label(body)
    if n_cc > 1:  # 只保留最大连通域
        body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)

    ggo = -620 + p["tex_sigma"] * smooth_noise(shape, 1.0 / sp, rng)
    vol = np.where(body, np.maximum(vol, ggo), vol)

    # 3) 实性成分（大小由 CTR 决定）
    if p["ctr_true"] > 0:
        cs = c + rng.normal(0, 0.15 * R * (1 - p["ctr_true"]) + 1e-3, 3).astype(np.float32)
        rho_s = np.linalg.norm((P - cs) / ax, axis=-1)
        solid = body & (rho_s < R * p["ctr_true"] * (1 + 0.5 * p["lob_amp"] * lob))
        vol = np.where(solid, 10 + 25 * smooth_noise(shape, 0.8 / sp, rng), vol)

    # 4) 毛刺：从结节表面向外放射的细线
    for _ in range(p["n_spikes"]):
        d = unit(rng).astype(np.float32)
        w = soft_tube(P, c + d * R * 0.8, d, rng.uniform(0.5, 0.9), length=rng.uniform(4, 10))
        vol = np.maximum(vol, -860 + w * 810)

    # 5) 空泡征：结节内部的小气腔
    if p["vacuole_true"]:
        cand = np.argwhere(body & (rho < 0.6 * R))
        if len(cand):
            vc = P[tuple(cand[rng.integers(len(cand))])]
            vol[np.linalg.norm(P - vc, axis=-1) < rng.uniform(1.0, 1.6)] = -900

    # 6) 部分容积效应 + CT 噪声
    vol = ndimage.gaussian_filter(vol, 0.6)
    vol = vol + noise * smooth_noise(shape, 0.7, rng)
    vol = np.clip(vol, -1024, 3071).astype(np.int16)

    # 7) 模拟医生勾画：随机外扩/内缩一圈
    mask = body.copy()
    r = rng.random()
    if r < 0.2:
        mask = ndimage.binary_dilation(mask)
    elif r < 0.4 and ndimage.binary_erosion(mask).sum() > 30:
        mask = ndimage.binary_erosion(mask)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return vol, mask


def _write_case(args):
    pid, p, scan, seed = args
    vol, mask = make_volume(p, scan["spacing"], scan["noise"], seed)
    utils.save_nifti(vol, scan["spacing"], config.IMAGE_DIR / f"{pid}.nii.gz")
    utils.save_nifti(mask, scan["spacing"], config.MASK_DIR / f"{pid}.nii.gz")
    return pid


def main():
    if config.CLINICAL_CSV.exists() and not config.DEMO_FLAG.exists():
        print(f"[停止] {config.CLINICAL_CSV} 已存在且不是模拟数据，为防止覆盖真实数据，不生成模拟数据。")
        return
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.MASK_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.SEED)
    rows, jobs = [], []
    for center, scan in CENTERS.items():
        for i in range(scan["n"]):
            pid = f"{center}{i + 1:03d}"
            p, clin = sample_patient(rng, scan["shift"])
            rows.append({config.ID_COL: pid, config.CENTER_COL: center, **clin})
            jobs.append((pid, p, scan, int(rng.integers(1 << 31))))

    print(f"生成 {len(jobs)} 个模拟病例（{config.N_WORKERS} 个进程并行）...")
    with ProcessPoolExecutor(config.N_WORKERS) as ex:
        for k, _ in enumerate(ex.map(_write_case, jobs, chunksize=4), 1):
            if k % 60 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}")

    df = pd.DataFrame(rows)
    df.to_csv(config.CLINICAL_CSV, index=False, encoding="utf-8-sig")
    config.DEMO_FLAG.write_text(
        "data/raw 中的数据由 src/00_make_demo_data.py 随机生成，仅用于调试代码，不是真实病人数据。\n"
        "换成真实数据时：删除整个 data/raw 文件夹内容（包括本文件），再放入真实数据。\n",
        encoding="utf-8")
    print(df.groupby(config.CENTER_COL)[config.LABEL_COL].agg(["count", "mean"])
            .rename(columns={"count": "n", "mean": "IAC_ratio"}))
    print(f"临床表已保存：{config.CLINICAL_CSV}")


if __name__ == "__main__":
    main()

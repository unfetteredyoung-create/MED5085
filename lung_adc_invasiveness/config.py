"""
全局配置：路径、变量定义、超参数都集中在这里。换数据 / 调参数时只改这个文件。

做实验时不要直接改这里的默认值，而是新建 experiments/<实验名>.json 写要改的参数，
再用 python run_all.py --exp <实验名> 运行（见 docs/优化迭代路线.md）。文件末尾会读取并覆盖。
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------- 路径 ----------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"           # 每个病人一个术前 CT：<patient_id>.nii.gz
MASK_DIR = RAW_DIR / "masks"             # 每个病人一个结节勾画（ROI）：<patient_id>.nii.gz，结节=1，背景=0
MASK2_DIR = RAW_DIR / "masks_reader2"    # （可选）第二位医生对部分病例的勾画，用于 ICC 稳定性分析
CLINICAL_CSV = RAW_DIR / "clinical.csv"  # 临床表，一行一个病人
DEMO_FLAG = RAW_DIR / "SIMULATED_DATA.txt"   # 有这个文件 = 当前是模拟数据，所有图都会打上水印

PROCESSED_DIR = DATA_DIR / "processed"
PATCH_DIR = PROCESSED_DIR / "patches"
META_CSV = PROCESSED_DIR / "meta.csv"    # 临床表 + 队列划分（train / internal_test / external_test）

SEED = 42
FAST = os.environ.get("FAST", "0") == "1"   # FAST=1：快速冒烟测试（少训练几轮、少提特征）

# ---------------- 标签与队列 ----------------
ID_COL = "patient_id"
LABEL_COL = "label"          # 0 = AIS/MIA（原位/微浸润），1 = IAC（浸润性腺癌）
CENTER_COL = "center"
EXTERNAL_CENTERS = ["B"]     # 这些中心的病人整体作为外部验证集；单中心数据就改成 []
INTERNAL_TEST_SIZE = 0.3     # 剩余病人按 7:3 分层随机划分为训练集 / 内部测试集
CV_FOLDS = 5
COHORTS = ["train", "internal_test", "external_test"]

# ---------------- 临床 + CT 语义特征 ----------------
# "num" = 连续变量，"bin" = 二分类变量 (0/1)
CLINICAL_VARS = {
    "age": "num",
    "sex": "bin",                  # 1 = 男
    "smoking": "bin",              # 吸烟史
    "cea": "num",                  # 血清 CEA, ng/mL
    "diameter_mm": "num",          # CT 上结节最大径
    "ctr": "num",                  # 实性成分占比 consolidation-to-tumor ratio，百分比 0-100
    "lobulation": "bin",           # 分叶征
    "spiculation": "bin",          # 毛刺征
    "pleural_indentation": "bin",  # 胸膜凹陷征
    "vacuole": "bin",              # 空泡征
}
UNIVARIATE_P = 0.05    # 单因素 p < 0.05 的变量进入多因素
MULTIVARIATE_P = 0.05  # 多因素向后剔除，保留 p < 0.05 的独立危险因素

DISPLAY_NAMES = {
    "age": "Age (years)",
    "sex": "Male sex",
    "smoking": "Smoking history",
    "cea": "CEA (ng/mL)",
    "diameter_mm": "Diameter (mm)",
    "ctr": "CTR (%)",
    "lobulation": "Lobulation",
    "spiculation": "Spiculation",
    "pleural_indentation": "Pleural indentation",
    "vacuole": "Vacuole sign",
    "rad_score": "Rad-score",
    "dl_score": "DL-score",
}

# ---------------- 影像预处理 ----------------
TARGET_SPACING = (1.0, 1.0, 1.0)   # 统一重采样到 1 mm 各向同性
CROP_SIZE = 72                      # 以结节为中心存 72^3 的块；训练时随机裁 64^3 做平移增强
PATCH_SIZE = 64
HU_WINDOW = (-1350, 150)            # 肺窗：WL -600 / WW 1500

# ---------------- 影像组学 (PyRadiomics) ----------------
RADIOMICS_SETTINGS = {
    "binWidth": 25,
    "normalize": False,
    "force2D": False,
    "label": 1,
}
RADIOMICS_IMAGE_TYPES = {
    "Original": {},
    "LoG": {"sigma": [1.0, 2.0, 3.0]},
    "Wavelet": {},
}
if FAST:
    RADIOMICS_IMAGE_TYPES = {"Original": {}, "LoG": {"sigma": [2.0]}}
PERITUMORAL_MM = 0                  # >0：额外提取结节外扩 N mm 环形区域（瘤周）的特征，前缀 peri_
ICC_N_CASES = 10 if FAST else 30    # 用多少个训练病例做勾画稳定性（ICC）分析
ICC_THRESHOLD = 0.75                # 只保留 ICC >= 0.75 的稳定特征
MWU_P = 0.05                        # Mann-Whitney U 检验筛选阈值
CORR_THRESHOLD = 0.9                # 两两相关 |r| > 0.9 只保留一个
LASSO_ONE_SE = True                 # True：1-SE 规则（特征更少、更稳）；False：交叉验证误差最小的 λ
N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

# ---------------- 深度学习 (3D CNN) ----------------
DL_BACKBONE = "resnet3d"     # "resnet3d"：小型 3D ResNet 从头训练；"resnet18_2.5d"：三切面 + ImageNet 预训练
DL_EPOCHS = 6 if FAST else 40
DL_BATCH = 16
DL_LR = 1e-3
DL_WEIGHT_DECAY = 1e-4
DL_DROPOUT = 0.3
DL_VARIANTS = ["image", "image_clinical"]   # 纯影像 CNN；影像+临床端到端融合网络

# ---------------- 实验管理 ----------------
# 用 experiments/<EXP_NAME>.json 覆盖上面的参数（键名 = 变量名，以 _ 开头的键是注释）
EXP_NAME = os.environ.get("EXP", "v0_baseline")
EXP_OVERRIDES = {}
_exp_file = ROOT / "experiments" / f"{EXP_NAME}.json"
if _exp_file.exists():
    EXP_OVERRIDES = {k: v for k, v in json.loads(_exp_file.read_text(encoding="utf-8")).items()
                     if not k.startswith("_")}
    _unknown = [k for k in EXP_OVERRIDES if k not in globals()]
    if _unknown:
        raise KeyError(f"{_exp_file.name} 里有 config.py 中不存在的参数：{_unknown}")
    globals().update(EXP_OVERRIDES)

# 每个实验的输出单独放：outputs/<实验名>/
OUT_DIR = ROOT / "outputs" / EXP_NAME
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"

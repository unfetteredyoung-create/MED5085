# 早期肺腺癌病理侵袭性预测：术前 CT 深度学习影像组学 + 临床特征

MED5085 多模态医学数据分析与人工智能 · Fall 2026 · 课程项目代码

> **当前状态：用模拟数据把整条流程跑通了。** `data/raw/` 里的 CT、勾画和临床表都是程序随机生成的，
> 所有图都带 “SIMULATED DATA” 水印，结果**不能**写进报告当真实结论。拿到真实数据后按
> [换成真实数据](#换成真实数据) 一节了解输入格式；真实数据研究前须修复验证设计和影像空间核验等问题，详见代码审阅文档。
>
> 文档：
> - [docs/项目方案.md](docs/项目方案.md)：课题背景、技术路线、分工、时间安排（给全组看）
> - [docs/上手教程.md](docs/上手教程.md)：编程手从跑通到看懂到改得动（**新手从这里开始**）
> - [docs/代码项目小白导览.md](docs/代码项目小白导览.md)：逐层解释 v0/v1 代码关系、七步流水线和第一次动手路线（**完全零基础先看这份**）
> - [docs/优化迭代路线.md](docs/优化迭代路线.md)：怎么做实验、优化方向全景、v1 迭代结果

---

## 快速开始

用课程统一的 conda 环境（python 3.9 + torch 2.0.1 + CUDA 11.7，见课程文件 `环境配置.txt`），再补装几个包：

```bash
pip install -r requirements.txt --only-binary=pyradiomics
```

一键运行（生成模拟数据 → 预处理 → 影像组学 → 临床模型 → 深度学习 → 融合与评价）：

```bash
python run_all.py
```

其他用法：

```bash
python run_all.py --fast      # 冒烟测试：少训练几轮、少提特征，几分钟跑完
python run_all.py --from 5    # 从第 5 步开始重跑（前面的结果已经存在）
python run_all.py --real      # 真实数据：跳过生成模拟数据

# 做实验：experiments/<实验名>.json 写要改的参数，以已有实验为基础只重跑受影响的步骤
python run_all.py --exp v1a_peritumoral --base v0_baseline --steps 2,4,6
python compare_experiments.py # 所有实验的排行榜
```

本机（RTX 4060 笔记本）完整跑一遍约 15 分钟（深度学习占 13 分钟）；没有 GPU 也能跑，深度学习那一步会慢很多，可先用 `--fast`。
每个实验的结果单独放在 `outputs/<实验名>/`，默认实验名是 `v0_baseline`。
学校集群上用 `bsub < run_hpc.lsf` 提交（队列名按集群手册改）。

## 项目结构

```
lung_adc_invasiveness/
├── config.py                 # 所有路径、变量、超参数的默认值
├── experiments/              # 一个实验一个 json：相对默认值改了哪些参数
├── run_all.py                # 一键运行（--exp / --base / --steps）
├── compare_experiments.py    # 汇总所有实验 → outputs/leaderboard.csv / .png
├── run_hpc.lsf               # 集群作业脚本
├── pretrained/               # 自动下载的 ImageNet 预训练权重（放在项目里，不占 C 盘）
├── src/
│   ├── utils.py              # 公共函数：读写 NIfTI、DeLong、DCA、画图…
│   ├── 00_make_demo_data.py  # 生成模拟数据（真实数据不用跑）
│   ├── 01_preprocess.py      # 重采样 1mm、以结节为中心裁 72³、划分队列、QC 图
│   ├── 02_radiomics_extract.py  # PyRadiomics 提特征 + ICC 稳定性
│   ├── 03_clinical_model.py  # Table 1、单/多因素 logistic、临床模型
│   ├── 04_radiomics_model.py # 特征筛选漏斗 + LASSO → Rad-score
│   ├── 05_deep_learning.py   # 3D ResNet（纯影像 / 影像+临床端到端融合）+ Grad-CAM
│   └── 06_fusion_evaluate.py # 列线图、晚期融合、ROC/DeLong/校准/DCA/SHAP
├── data/
│   ├── raw/                  # 原始数据：images/、masks/、clinical.csv
│   └── processed/            # 预处理结果：patches/*.npz、meta.csv（含队列划分）
├── outputs/
│   ├── leaderboard.csv/.png  # 实验排行榜
│   └── <实验名>/              # 如 v0_baseline/
│       ├── figures/          # 所有图（可直接放 PPT）
│       ├── tables/           # 所有表（csv，Excel 可直接打开）
│       ├── models/           # 训练好的模型
│       ├── results_summary.md  # 自动生成的结果汇总
│       ├── experiment.json   # 实验档案：改了什么、跑过哪些步骤
│       └── run.log           # 完整运行日志
└── docs/                     # 项目方案、上手教程、优化迭代路线
```

## 每一步做什么、产出什么

| 步骤 | 做什么 | 主要产出 |
|---|---|---|
| 0 | 生成 360 个模拟病人（中心 A 260 例薄层 CT；中心 B 100 例厚层、噪声大，作外部验证） | `data/raw/` |
| 1 | 重采样到 1×1×1 mm；以结节外接框中心裁 72³ 块；中心 B → 外部验证，中心 A 按 7:3 分层随机分训练/内部测试 | `01_qc_patches.png`、`meta.csv` |
| 2 | PyRadiomics：原图 + LoG(σ=1,2,3) + 小波，共 1130 个特征；30 例双勾画算 ICC | `radiomics_features.csv`、`02_radiomics_icc.png` |
| 3 | Table 1；单因素 → 多因素 logistic（向后剔除）找独立危险因素；临床模型 | `table1_baseline.csv`、`clinical_logistic_regression.csv` |
| 4 | ICC ≥ 0.75 → Mann-Whitney U → 去相关 \|r\|>0.9 → LASSO(1-SE) → Rad-score | `04_lasso.png`、`04_rad_signature.png`、`radiomics_signature.csv` |
| 5 | 小型 3D ResNet，训练集 5 折交叉验证；两个版本：纯影像 / 影像+临床中间层融合 | `05_learning_curve_*.png`、`05_grad_cam.png` |
| 6 | 列线图（临床 + Rad-score + DL-score）、晚期融合；全部模型在三个队列上评价 | `06_roc_curves.png`、`06_calibration.png`、`06_decision_curves.png`、`06_nomogram.png`、`06_shap_nomogram.png`、`performance.csv`、`delong_tests.csv` |

**防止数据泄露**：所有筛选和拟合只用训练集；训练集上的 Rad-score / DL-score 是 5 折“折外预测”，
所以融合模型不会“自己考自己”；测试集和外部验证集全程只用来算分。

## Demo 结果（模拟数据，仅验证流程）

基线 `v0_baseline` 的 AUC（95% CI）。train 列是训练集 5 折交叉验证的折外预测：

| 模型 | train（5 折 CV） | 内部测试 (n=78) | 外部验证 (n=100) |
|---|---|---|---|
| Clinical | 0.855 (0.798-0.913) | 0.769 (0.663-0.876) | 0.847 (0.772-0.922) |
| Radiomics | 0.803 (0.738-0.868) | 0.782 (0.680-0.883) | 0.818 (0.730-0.905) |
| DL (image) | 0.768 (0.698-0.838) | 0.701 (0.584-0.817) | 0.792 (0.704-0.880) |
| DL (image+clinical) | 0.834 (0.774-0.894) | 0.753 (0.646-0.860) | 0.823 (0.742-0.904) |
| Late fusion | 0.852 (0.795-0.910) | 0.780 (0.678-0.883) | 0.841 (0.764-0.918) |
| Nomogram | 0.864 (0.808-0.920) | 0.757 (0.650-0.864) | 0.845 (0.770-0.919) |

v1 迭代实验的结果见 [docs/优化迭代路线.md](docs/优化迭代路线.md) 和 `outputs/leaderboard.png`。

这些数字**没有医学意义**（模拟数据的规律是人为设定的），但流程上的几个现象在真实研究里也常见、值得讨论：
- 内部测试集只有 78 人，AUC 的 95% CI 宽度约 0.2，模型之间的差别大多没有统计学意义（见 `delong_tests.csv`）。
- Rad-score 和 CTR 高度相关（r ≈ 0.79），放进列线图后不再独立显著——入选的影像组学特征主要在重复度量“实性成分”。
- 从头训练的 3D CNN 在小样本上弱于影像组学，加入临床分支后明显提升。

## 换成真实数据

1. **清空 `data/raw/`**（包括 `SIMULATED_DATA.txt`，这个文件在就会打水印），以及 `data/processed/`、`outputs/`。
2. **CT 与勾画**：每个病人一个 NIfTI 文件，文件名就是病人编号：
   - `data/raw/images/<patient_id>.nii.gz`：术前薄层 CT，单位 HU
   - `data/raw/masks/<patient_id>.nii.gz`：同尺寸的结节勾画，结节 = 1，背景 = 0
   - DICOM 转 NIfTI：`dcm2niix` 或 3D Slicer；勾画：ITK-SNAP 或 3D Slicer（导出时选和 CT 同一几何空间）
   - （可选）`data/raw/masks_reader2/`：第二位医生对 30 例左右的独立勾画，用于 ICC；没有就自动用扰动模拟
3. **临床表** `data/raw/clinical.csv`，一行一个病人：

   | 列名 | 含义 | 取值 |
   |---|---|---|
   | `patient_id` | 病人编号，与 NIfTI 文件名一致 | 字符串 |
   | `center` | 来源医院 | 如 A / B；单中心可全填 A |
   | `label` | 病理标签 | 0 = AIS/MIA，1 = IAC |
   | `age` | 年龄 | 岁 |
   | `sex` | 性别 | 1 = 男，0 = 女 |
   | `smoking` | 吸烟史 | 1 / 0 |
   | `cea` | 血清 CEA | ng/mL，缺失留空 |
   | `diameter_mm` | CT 最大径 | mm |
   | `ctr` | 实性成分占比（实性最大径 / 结节最大径） | 百分比 0–100 |
   | `lobulation` / `spiculation` / `pleural_indentation` / `vacuole` | 分叶 / 毛刺 / 胸膜凹陷 / 空泡征 | 1 / 0 |

   想增删变量：改 `config.py` 里的 `CLINICAL_VARS`。
4. **队列划分**：多中心时把作外部验证的中心写进 `config.EXTERNAL_CENTERS`；单中心改成 `[]`
   （那就只有训练集 + 内部测试集）。
5. 运行 `python run_all.py --real`。先看 `outputs/figures/01_qc_patches.png` 确认勾画和 CT 对得上。

## 常见问题

- **路径里有中文**：已处理。NIfTI 用 nibabel 读，PyRadiomics 在内存中转换，torch 模型通过文件句柄保存
  （torch 2.0 在 Windows 上直接传中文路径会报 “Parent directory does not exist”）。
- **pyradiomics 装不上**：Windows 上加 `--only-binary=pyradiomics` 用预编译包，不要从源码编译。
- **结节大于 72 mm**：预处理会打印警告，调大 `config.CROP_SIZE`。
- **显存不够**：调小 `config.DL_BATCH`。
- **结果和别人跑的不完全一样**：GPU 上的 cuDNN 有少量非确定性，深度学习结果会有小幅波动，属正常。

## 下一步（编程这边）

- [ ] 拿到真实数据后先跑 `--real`，看 QC 图和 Table 1
- [x] v1：实验管理、瘤周影像组学、2.5D 预训练网络、LASSO λ 规则对比、Grad-CAM 对齐修复
  （模拟数据上三个改动都没有带来有意义的提升，默认保持 v0；详见 [优化迭代路线](docs/优化迭代路线.md)）
- [ ] v2 及以后的方向见 [docs/优化迭代路线.md](docs/优化迭代路线.md) 第 3、6 节

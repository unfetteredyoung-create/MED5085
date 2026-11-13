"""
汇总 outputs/ 下所有实验的结果，生成排行榜：
    outputs/leaderboard.csv   每个实验 × 每个模型一行
    outputs/leaderboard.png   各实验训练集交叉验证 AUC 对比图

挑方案只看 train_CV_AUC（训练集 5 折折外预测，测试集没参与）；
测试集 AUC 只在阶段性节点参考，不要拿来调参。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import utils  # noqa: E402
from utils import plt  # noqa: E402

OUT = ROOT / "outputs"


def collect():
    rows = []
    for perf_file in sorted(OUT.glob("*/tables/performance.csv")):
        exp_dir = perf_file.parents[1]
        info_file = exp_dir / "experiment.json"
        info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
        if any(r.get("fast") for r in info.get("runs", [])):
            print(f"（跳过 {exp_dir.name}：--fast 冒烟测试的结果不参与比较）")
            continue
        perf = pd.read_csv(perf_file)
        for model, g in perf.groupby("Model", sort=False):
            auc = g.set_index("Cohort")["AUC"]
            rows.append({
                "experiment": exp_dir.name, "model": model,
                "train_CV_AUC": auc.get("train"),
                "internal_test_AUC": auc.get("internal_test"),
                "external_test_AUC": auc.get("external_test"),
                "changes": json.dumps(info.get("overrides", {}), ensure_ascii=False),
                "note": info.get("note", ""),
            })
    return pd.DataFrame(rows)


def plot(lb, exps, models):
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 3.8))
    w = 0.8 / len(exps)
    for j, exp in enumerate(exps):
        v = lb[lb["experiment"] == exp].set_index("model").reindex(models)["train_CV_AUC"]
        x = np.arange(len(models)) + (j - (len(exps) - 1) / 2) * w
        ax.bar(x, v, w * 0.92, label=exp, color=plt.cm.tab10(j % 10))
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=15, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Train 5-fold CV AUC")
    ax.set_title("Experiment leaderboard (selection metric: train CV AUC)")
    ax.legend(fontsize=7.5, frameon=False, ncol=min(4, len(exps)), loc="upper left")
    fig.tight_layout()
    utils.save_fig(fig, "leaderboard.png", folder=OUT)


def main():
    lb = collect()
    if lb.empty:
        print("outputs/ 下还没有任何实验结果（需要先跑完 Step 6）")
        return
    lb.to_csv(OUT / "leaderboard.csv", index=False, encoding="utf-8-sig")
    exps, models = list(lb["experiment"].unique()), list(lb["model"].unique())

    pd.set_option("display.width", 200)
    for metric, title in [("train_CV_AUC", "训练集 5 折交叉验证 AUC（用这个挑方案）"),
                          ("internal_test_AUC", "内部测试集 AUC（只在阶段节点参考）")]:
        pv = lb.pivot(index="model", columns="experiment", values=metric).reindex(index=models, columns=exps)
        print(f"\n{title}\n{pv.round(3).to_string()}")

    best = lb.loc[lb.groupby("model", sort=False)["train_CV_AUC"].idxmax(), ["model", "experiment", "train_CV_AUC"]]
    print(f"\n每个模型交叉验证 AUC 最高的实验：\n{best.round(3).to_string(index=False)}")
    for exp in exps:
        note = lb.loc[lb["experiment"] == exp, "note"].iloc[0]
        print(f"  {exp}: {note}")
    plot(lb, exps, models)
    print(f"\n排行榜：{OUT / 'leaderboard.csv'}，{OUT / 'leaderboard.png'}")


if __name__ == "__main__":
    main()

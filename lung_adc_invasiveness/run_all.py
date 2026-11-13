"""
一键运行整条流程。每个实验的结果单独放在 outputs/<实验名>/。

用法：
    python run_all.py                       # 默认实验 v0_baseline：生成模拟数据 + 跑完全部步骤
    python run_all.py --fast                # 快速冒烟测试：少训练几轮、少提特征（结果放 outputs/smoke_fast/）
    python run_all.py --from 4              # 从第 4 步开始（前面的结果已经有了）
    python run_all.py --real                # 真实数据：跳过 Step 0（生成模拟数据）

    # 迭代优化：experiments/<实验名>.json 里写要改的参数，以已有实验为基础只重跑受影响的步骤
    python run_all.py --exp v1a_peritumoral --base v0_baseline --steps 2,4,6
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "00_make_demo_data.py",
    "01_preprocess.py",
    "02_radiomics_extract.py",
    "03_clinical_model.py",
    "04_radiomics_model.py",
    "05_deep_learning.py",
    "06_fusion_evaluate.py",
]


def run_step(i, env, log):
    script = STEPS[i]
    banner = f"\n{'=' * 70}\n[Step {i}] {script}  (实验 {env['EXP']})\n{'=' * 70}\n"
    print(banner, end="", flush=True)
    log.write(banner)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:   # 同时打印到屏幕和写进日志
        print(line, end="", flush=True)
        log.write(line)
    if proc.wait() != 0:
        sys.exit(f"[Step {i}] 失败（退出码 {proc.returncode}），修好后可用 --steps 或 --from 从这一步继续")
    msg = f"[Step {i}] 完成，用时 {time.time() - t0:.0f}s\n"
    print(msg, end="", flush=True)
    log.write(msg)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", help="实验名，对应 experiments/<实验名>.json（可以不存在）；"
                                  "默认 v0_baseline，--fast 时默认 smoke_fast（不覆盖正式结果）")
    ap.add_argument("--base", help="以哪个已有实验为基础（先把它的结果复制过来）")
    ap.add_argument("--steps", help="只跑这些步骤，如 2,4,6")
    ap.add_argument("--from", dest="start", type=int, default=0, help="从第几步开始 (0-6)")
    ap.add_argument("--fast", action="store_true", help="快速冒烟测试")
    ap.add_argument("--real", action="store_true", help="使用真实数据，跳过 Step 0")
    args = ap.parse_args()
    args.exp = args.exp or ("smoke_fast" if args.fast else "v0_baseline")

    exp_file = ROOT / "experiments" / f"{args.exp}.json"
    exp_cfg = json.loads(exp_file.read_text(encoding="utf-8")) if exp_file.exists() else {}
    out_dir = ROOT / "outputs" / args.exp

    if args.base:
        base_dir = ROOT / "outputs" / args.base
        if not base_dir.exists():
            sys.exit(f"基础实验 {base_dir} 不存在")
        print(f"从 {args.base} 复制已有结果到 {args.exp} ...")
        shutil.copytree(base_dir, out_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("experiment.json", "run.log", "results_summary.md"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.steps:
        steps = sorted(int(s) for s in args.steps.split(","))
    else:
        steps = [i for i in range(len(STEPS)) if i >= max(args.start, 1 if args.real else 0)]
    if args.base and 0 in steps:
        sys.exit("以已有实验为基础时不要重新生成数据（去掉 Step 0）")

    info_file = out_dir / "experiment.json"   # 实验档案：改了什么 + 每次运行了哪些步骤
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    info.update(name=args.exp, note=exp_cfg.get("_note", ""),
                overrides={k: v for k, v in exp_cfg.items() if not k.startswith("_")})
    info.setdefault("runs", []).append({"base": args.base, "steps": steps, "fast": args.fast,
                                        "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    env = {**os.environ, "EXP": args.exp, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "FAST": "1" if args.fast else "0"}
    t_all = time.time()
    with open(out_dir / "run.log", "a", encoding="utf-8", buffering=1) as log:   # 行缓冲：运行中也能看日志
        for i in steps:
            run_step(i, env, log)
    print(f"\n全部完成，总用时 {(time.time() - t_all) / 60:.1f} 分钟。结果在 outputs/{args.exp}/，"
          f"汇总见 outputs/{args.exp}/results_summary.md\n各实验对比：python compare_experiments.py")


if __name__ == "__main__":
    main()

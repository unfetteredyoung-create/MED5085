"""
Step 5：深度学习（3D CNN）

两个模型（config.DL_VARIANTS）：
  image          ：只看 CT 结节块的 3D ResNet，输出 DL-score
  image_clinical ：3D ResNet 的影像特征 + 临床特征 MLP，在网络中间层拼接（端到端的中间层融合）

训练方式：训练集上 5 折交叉验证
  - 训练集病人的预测 = 折外预测（由没见过他的那一折模型给出），后面做融合时不会“自己考自己”
  - 测试集病人的预测 = 5 个折模型的平均
另外输出训练曲线和 Grad-CAM 热图（看模型关注 CT 的哪个区域）。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# 预训练权重下载到项目里的 pretrained/（F 盘），而不是默认的 C:\Users\...\.cache
os.environ.setdefault("TORCH_HOME", str(config.ROOT / "pretrained"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import utils  # noqa: E402
from utils import plt  # noqa: E402

Y = config.LABEL_COL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"   # 混合精度：更快、更省显存
torch.backends.cudnn.benchmark = True


# ======================= 数据 =======================
def load_patches(pids):
    imgs = [utils.window_normalize(np.load(config.PATCH_DIR / f"{p}.npz")["image"]) for p in pids]
    return np.stack(imgs).astype(np.float16)   # float16 存内存，省一半


def center_crop(img):
    o = (config.CROP_SIZE - config.PATCH_SIZE) // 2
    s = config.PATCH_SIZE
    return img[..., o:o + s, o:o + s, o:o + s]


class NoduleDataset(Dataset):
    def __init__(self, images, labels, clin, train):
        self.images, self.labels, self.clin, self.train = images, labels, clin, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.train:  # 数据增强：随机平移裁剪 + 三个方向随机翻转 + 轴位面随机旋转 90°
            s = config.PATCH_SIZE
            o = np.random.randint(0, config.CROP_SIZE - s + 1, 3)
            img = img[o[0]:o[0] + s, o[1]:o[1] + s, o[2]:o[2] + s]
            for ax in range(3):
                if np.random.rand() < 0.5:
                    img = np.flip(img, ax)
            img = np.rot90(img, np.random.randint(4), axes=(0, 1))
        else:
            img = center_crop(img)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None]
        return x, torch.from_numpy(self.clin[i]), torch.tensor(self.labels[i], dtype=torch.float32)


# ======================= 模型 =======================
class BasicBlock3d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def add_fusion_head(module, n_feat, n_clin, dropout):
    """影像特征 (n_feat 维) 后面接分类头；n_clin > 0 时多一个临床分支，两者拼接后再分类（中间层融合）。"""
    module.clin_mlp = None
    if n_clin:
        module.clin_mlp = nn.Sequential(nn.Linear(n_clin, 32), nn.ReLU(inplace=True), nn.Dropout(dropout))
        n_feat += 32
    module.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, 1))


def fusion_forward(module, h, clin):
    if module.clin_mlp is not None:
        h = torch.cat([h, module.clin_mlp(clin)], dim=1)
    return module.head(h).squeeze(1)


class ResNet3D(nn.Module):
    """小型 3D ResNet：64^3 → 32^3 → 16^3 → 8^3，约 1M 参数（样本少，网络不宜太大）。"""

    def __init__(self, n_clin=0, widths=(16, 32, 64, 128), dropout=config.DL_DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm3d(widths[0]), nn.ReLU(inplace=True))
        self.layer1 = BasicBlock3d(widths[0], widths[1], 2)
        self.layer2 = BasicBlock3d(widths[1], widths[2], 2)
        self.layer3 = BasicBlock3d(widths[2], widths[3], 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        add_fusion_head(self, widths[3], n_clin, dropout)

    def forward(self, x, clin=None):
        h = self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).flatten(1)
        return fusion_forward(self, h, clin)


class ResNet2p5D(nn.Module):
    """2.5D：取结节中心的轴位、冠状位、矢状位三个切面，当成 RGB 三个通道，
    输入 ImageNet 预训练的 ResNet18（和 Lab3 一样的迁移学习思路，小样本时通常更稳）。"""

    def __init__(self, n_clin=0, dropout=config.DL_DROPOUT, size=128):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)   # 第一次运行会下载约 45 MB 权重
        self.backbone.fc = nn.Identity()
        self.size = size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        add_fusion_head(self, 512, n_clin, dropout)

    def forward(self, x, clin=None):           # x: (B, 1, X, Y, Z)
        c = x.shape[-1] // 2
        planes = torch.cat([x[..., c],              # 轴位
                            x[:, :, :, c, :],       # 冠状位
                            x[:, :, c, :, :]], 1)   # 矢状位 → (B, 3, 64, 64)
        planes = F.interpolate(planes, size=(self.size, self.size), mode="bilinear", align_corners=False)
        h = self.backbone((planes - self.mean) / self.std)
        return fusion_forward(self, h, clin)


def build_model(n_clin):
    if config.DL_BACKBONE == "resnet3d":
        return ResNet3D(n_clin)
    if config.DL_BACKBONE == "resnet18_2.5d":
        return ResNet2p5D(n_clin)
    raise ValueError(f"未知的 DL_BACKBONE：{config.DL_BACKBONE}")


# ======================= 训练 / 预测 =======================
@torch.no_grad()
def predict(model, images, clin, batch=32):
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(center_crop(images[i:i + batch]).astype(np.float32))[:, None].to(DEVICE)
        c = torch.from_numpy(clin[i:i + batch]).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            out.append(model(x, c).float().cpu().numpy())
    return np.concatenate(out)


def train_fold(img_tr, y_tr, c_tr, img_va, y_va, c_va, n_clin):
    model = build_model(n_clin).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=config.DL_LR, weight_decay=config.DL_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.DL_EPOCHS)
    pos_weight = torch.tensor((y_tr == 0).sum() / max(1, (y_tr == 1).sum()), dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # 类别不平衡时给少数类加权
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    loader = DataLoader(NoduleDataset(img_tr, y_tr, c_tr, train=True), batch_size=config.DL_BATCH,
                        shuffle=True, drop_last=True, num_workers=0)

    history = []
    for epoch in range(config.DL_EPOCHS):
        model.train()
        total, n = 0.0, 0
        for x, c, y in loader:
            x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                loss = criterion(model(x, c), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        p_va = utils.sigmoid(predict(model, img_va, c_va))
        history.append({"epoch": epoch + 1, "train_loss": total / n,
                        "val_loss": log_loss(y_va, p_va, labels=[0, 1]), "val_auc": roc_auc_score(y_va, p_va)})
    return model, history   # 固定训练轮数、取最后一轮，不用验证集挑 epoch，避免折外预测偏乐观


# ======================= Grad-CAM =======================
def grad_cam(model, img):
    """对 layer3（8^3 特征图）做 Grad-CAM，上采样回 64^3。"""
    feats = {}
    hook = model.layer3.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    model.eval()
    x = torch.from_numpy(center_crop(img).astype(np.float32))[None, None].to(DEVICE)
    with torch.enable_grad():
        logit = model(x)
        grads = torch.autograd.grad(logit.sum(), feats["a"])[0]
    hook.remove()
    a = feats["a"].detach()
    cam = F.relu((grads.mean(dim=(2, 3, 4), keepdim=True) * a).sum(1, keepdim=True))
    # 三次 stride-2 卷积后，第 j 个特征格对应输入第 8j 个体素：先插值到 0..8(n-1)，再在末端补齐，保证热图不偏移
    n, s = cam.shape[-1], x.shape[-1] // cam.shape[-1]
    cam = F.interpolate(cam, size=(s * (n - 1) + 1,) * 3, mode="trilinear", align_corners=True)
    pad = x.shape[-1] - cam.shape[-1]
    cam = F.pad(cam, (0, pad, 0, pad, 0, pad), mode="replicate")[0, 0]
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    return cam, float(torch.sigmoid(logit).item())


def plot_grad_cam(model, meta, images):
    test = meta[meta["cohort"] == "internal_test"]
    rng = np.random.default_rng(config.SEED)
    picks = list(rng.choice(test.index[test[Y] == 1], 3, replace=False)) + \
        list(rng.choice(test.index[test[Y] == 0], 3, replace=False))
    fig, axs = plt.subplots(2, len(picks), figsize=(2.3 * len(picks), 4.9))
    for k, i in enumerate(picks):
        pid = meta.at[i, config.ID_COL]
        mask = center_crop(np.load(config.PATCH_DIR / f"{pid}.npz")["mask"])
        cam, prob = grad_cam(model, images[i].astype(np.float32))
        z = int(np.argwhere(mask)[:, 2].mean())
        ct = center_crop(images[i].astype(np.float32))[:, :, z].T
        for row in range(2):
            ax = axs[row, k]
            ax.imshow(ct, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                ax.contour(mask[:, :, z].T, levels=[0.5], colors="#ff3b30", linewidths=0.7)
                ax.set_title(f"{pid} | {'IAC' if meta.at[i, Y] == 1 else 'AIS/MIA'}\nP(IAC)={prob:.2f}", fontsize=8)
            else:  # 只显示较强的关注区域，弱响应透明
                ax.imshow(np.ma.masked_less(cam[:, :, z].T, 0.2), cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.axis("off")
    fig.suptitle("Grad-CAM of the 3D CNN (top: CT + ROI, bottom: attention)", y=1.0)
    fig.tight_layout()
    utils.save_fig(fig, "05_grad_cam.png")


def plot_history(hist, variant):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for fold, h in hist.groupby("fold"):
        axs[0].plot(h["epoch"], h["train_loss"], color="#4C72B0", alpha=0.35, lw=1)
        axs[0].plot(h["epoch"], h["val_loss"], color="#C44E52", alpha=0.35, lw=1)
        axs[1].plot(h["epoch"], h["val_auc"], color="#55A868", alpha=0.35, lw=1)
    m = hist.groupby("epoch").mean(numeric_only=True)
    axs[0].plot(m.index, m["train_loss"], color="#4C72B0", lw=2, label="train")
    axs[0].plot(m.index, m["val_loss"], color="#C44E52", lw=2, label="validation fold")
    axs[1].plot(m.index, m["val_auc"], color="#55A868", lw=2)
    axs[0].set(xlabel="Epoch", ylabel="Loss", title=f"[{variant}] loss (thin = each fold)")
    axs[1].set(xlabel="Epoch", ylabel="AUC", title=f"[{variant}] validation AUC", ylim=(0.4, 1.0))
    axs[0].legend(frameon=False)
    fig.tight_layout()
    utils.save_fig(fig, f"05_learning_curve_{variant}.png")


# ======================= 主流程 =======================
def main():
    utils.ensure_dirs()
    utils.set_seed()
    print(f"设备：{DEVICE}{' (' + torch.cuda.get_device_name(0) + ')' if DEVICE.type == 'cuda' else ''}，"
          f"网络：{config.DL_BACKBONE}，每折训练 {config.DL_EPOCHS} 轮，学习率 {config.DL_LR}")
    meta = utils.impute_clinical(utils.load_meta())
    images = load_patches(meta[config.ID_COL])
    labels = meta[Y].values.astype(np.float32)
    clin_all = meta[list(config.CLINICAL_VARS)].values.astype(np.float32)
    tr_idx = np.where(meta["cohort"] == "train")[0]
    te_idx = np.where(meta["cohort"] != "train")[0]
    folds = list(StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
                 .split(tr_idx, labels[tr_idx]))

    for variant in config.DL_VARIANTS:
        use_clin = variant == "image_clinical"
        logits = np.zeros(len(meta))
        test_logits, histories = [], []
        t0 = time.time()
        for fold, (a, b) in enumerate(folds):
            utils.set_seed(config.SEED + fold)
            tr, va = tr_idx[a], tr_idx[b]
            if use_clin:  # 临床变量标准化只用这一折的训练部分 fit
                scaler = StandardScaler().fit(clin_all[tr])
                clin = scaler.transform(clin_all).astype(np.float32)
                joblib.dump(scaler, config.MODEL_DIR / f"dl_{variant}_fold{fold}_scaler.joblib")
            else:
                clin = np.zeros((len(meta), 0), dtype=np.float32)
            model, hist = train_fold(images[tr], labels[tr], clin[tr], images[va], labels[va], clin[va],
                                     n_clin=clin.shape[1])
            logits[va] = predict(model, images[va], clin[va])
            test_logits.append(predict(model, images[te_idx], clin[te_idx]))
            # 用 Python 文件句柄保存：torch 2.0 在 Windows 上直接传含中文的路径会报错
            with open(config.MODEL_DIR / f"dl_{variant}_fold{fold}.pt", "wb") as f:
                torch.save(model.state_dict(), f)
            histories += [{**h, "fold": fold} for h in hist]
            print(f"  [{variant}] fold {fold + 1}/{len(folds)}  val AUC = {hist[-1]['val_auc']:.3f}"
                  f"  ({time.time() - t0:.0f}s)")
            if fold == 0 and not use_clin and config.DL_BACKBONE == "resnet3d":  # Grad-CAM 目前只做了 3D 版
                plot_grad_cam(model, meta, images)
        logits[te_idx] = np.mean(test_logits, axis=0)

        hist = pd.DataFrame(histories)
        utils.save_table(hist, f"dl_history_{variant}.csv")
        plot_history(hist, variant)
        scores = meta[[config.ID_COL, "cohort", Y]].assign(dl_logit=logits, dl_prob=utils.sigmoid(logits))
        utils.save_table(scores, f"scores_dl_{variant}.csv")
        for c in config.COHORTS:
            s = scores[scores["cohort"] == c]
            if len(s):
                print(f"  DL[{variant}] AUC [{c}]: {utils.auc(s[Y], s['dl_prob']):.3f}")


def gradcam_only():
    """不重新训练，只用已保存的第 1 折纯影像模型重画 Grad-CAM：python src/05_deep_learning.py --gradcam-only"""
    meta = utils.impute_clinical(utils.load_meta())
    model = build_model(0).to(DEVICE)
    with open(config.MODEL_DIR / "dl_image_fold0.pt", "rb") as f:
        model.load_state_dict(torch.load(f, map_location=DEVICE))
    plot_grad_cam(model, meta, load_patches(meta[config.ID_COL]))


if __name__ == "__main__":
    gradcam_only() if "--gradcam-only" in sys.argv else main()

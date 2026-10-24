"""一期深度学习模型：元素嵌入 + 多任务共享主干 + 深度集成。

对应 docs/深度学习模型选用方案.md 的 DL-1。与现有 GBM 基线在同一套分组切分上对打。

读入
----
外部 Ms / Tc / Hcj 数据（三个任务的标签来源）

产出
----
data/32_dl_vs_gbm.csv        DL 集成与 GBM 逐任务、逐指标对比
data/32_dl_calibration.csv   80% 预测区间的实际覆盖率（不确定性校准）
data/32_dl_history.csv       训练曲线（每折每成员的早停轮次与验证损失）

口径
----
    - 三套外部数据按还原化学式合并成一张多任务表：每行一个材料，三个目标各自可缺失，
      损失用掩码，只在有标签的任务上回传梯度。这是多任务学习相对单任务 GBM 的核心增量。
    - 切分沿用项目口径：按化学体系分组 GroupKFold=5，同一体系不跨折。
    - GBM 对照在同一折上重训，保证"同切分、同评估"。
    - 深度集成给出预测方差，据此检验 80% 区间的实际覆盖率。
环境
----
    torch 装在 C:\\pylibs（Windows 长路径限制的绕行方案），脚本会自动把它加入 sys.path。
用法
----
python scripts/dl_multitask.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pymatgen.core import Composition

try:
    import torch
except ImportError:  # torch 装在短路径下
    sys.path.insert(0, r"C:\pylibs")
    import torch

import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import featlib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
BASE_EXT = r"C:\Users\Administrator\Desktop\02_Devs\Basis_Of_AI_Teamwork"
MS_EXT = os.path.join(BASE_EXT, r"Ms\outputs\Ms_cleaned.csv")
HCJ_EXT = os.path.join(BASE_EXT, r"Hcj\outputs\Hcj_cleaned.csv")
TC_EXT = os.path.join(BASE_EXT, r"Tc\tc_repm_deliverables\data\tc_dataset.csv")

TC_FLAGGED = {("Nd2Fe14B", 450.0), ("Nd2Fe14B", 768.15), ("Nd2Fe17", 743.0),
              ("Ce2Fe14B", 490.0), ("Er2Fe14B", 314.0), ("Y2Co17", 940.0)}

TASKS = ["Ms", "Tc", "Hcj"]
TASK_UNITS = {"Ms": "emu/g", "Tc": "K", "Hcj": "dex"}

RATIO_COLS = ["re_frac", "tm_frac", "b_frac", "re_over_metal", "b_over_re",
              "fe_over_re", "co_over_tm", "hre_over_re", "lre_over_re"]
WEIGHTED_COLS = ["w_mean_z", "w_mean_mass", "w_mean_radius", "w_mean_electroneg",
                 "w_mean_valence", "mix_entropy", "size_mismatch",
                 "electroneg_spread"]
MAGNETIC_COLS = ["re_avg_degennes", "re_avg_moment", "tm_avg_moment",
                 "has_re", "has_hre"]
EXTRA_COLS = ["is_oxide", "is_carbide"]
PHYS_COLS = RATIO_COLS + WEIGHTED_COLS + MAGNETIC_COLS + EXTRA_COLS

SEED = 0
N_MEMBERS_CV = 3
N_MEMBERS_FINAL = 8
MAX_EPOCHS = 200
PATIENCE = 20
BATCH = 128
EMB_DIM = 32


def reduced(f):
    try:
        return Composition(str(f)).reduced_formula
    except Exception:  # noqa: BLE001
        return None


def chem_sys(f):
    try:
        return "-".join(sorted(e.symbol for e in Composition(str(f)).elements))
    except Exception:  # noqa: BLE001
        return "UNPARSED"


def build_multitask_table() -> pd.DataFrame:
    """三套外部数据 → 一张多任务表（按还原化学式合并）。"""
    ms = pd.read_csv(MS_EXT, low_memory=False)
    ms = ms[(ms["Ms_conv_status"] == "ok") & ms["Ms_emu_g"].notna()]
    ms = (ms.groupby("material_name", as_index=False)
          .agg(Ms=("Ms_emu_g", "median")))
    ms["_red"] = ms["material_name"].map(reduced)

    tc = pd.read_csv(TC_EXT, low_memory=False)
    tc = tc[~tc.apply(lambda r: (str(r["formula"]), float(r["tc_K"])) in TC_FLAGGED,
                      axis=1)]
    tc = tc[tc["tc_K"] >= 10]
    tc = tc.groupby("formula", as_index=False).agg(Tc=("tc_K", "median"))
    tc["_red"] = tc["formula"].map(reduced)

    hc = pd.read_csv(HCJ_EXT, low_memory=False)
    hc = hc[hc["hcj_value_class"] == "plausible_hcj"]
    hc = hc.groupby("material_name", as_index=False).agg(v=("Hcj_kA_m", "median"))
    hc = hc[hc["v"].notna()]
    hc["Hcj"] = np.log10(1.0 + hc["v"].clip(lower=0))
    hc["_red"] = hc["material_name"].map(reduced)

    frames = []
    for d, col in ((ms, "Ms"), (tc, "Tc"), (hc, "Hcj")):
        d = d.dropna(subset=["_red"])[[col, "_red"]]
        frames.append(d.groupby("_red", as_index=False)[col].median())
    tab = frames[0]
    for nxt in frames[1:]:
        tab = tab.merge(nxt, on="_red", how="outer")
    tab = tab[tab[TASKS].notna().any(axis=1)].reset_index(drop=True)
    tab["chem_sys"] = tab["_red"].map(chem_sys)
    tab = tab[tab["chem_sys"] != "UNPARSED"].reset_index(drop=True)
    return tab


def element_vocab(formulas) -> list:
    els = set()
    for f in formulas:
        try:
            els |= {e.symbol for e in Composition(str(f)).elements}
        except Exception:  # noqa: BLE001
            continue
    return sorted(els)


def frac_matrix(formulas, vocab) -> np.ndarray:
    idx = {e: i for i, e in enumerate(vocab)}
    out = np.zeros((len(formulas), len(vocab)), dtype=np.float32)
    for r, f in enumerate(formulas):
        try:
            comp = Composition(str(f))
        except Exception:  # noqa: BLE001
            continue
        for el, x in comp.get_el_amt_dict().items():
            if el in idx:
                out[r, idx[el]] = x
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


class MultiTaskNet(nn.Module):
    """元素嵌入池化 ⊕ 物理描述符 → 共享主干 → 三个任务头。"""

    def __init__(self, n_elem: int, n_phys: int, emb: int = EMB_DIM):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(n_elem, emb) * 0.1)
        self.trunk = nn.Sequential(
            nn.Linear(2 * emb + n_phys, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in TASKS])
        self.log_scale = nn.Parameter(torch.zeros(len(TASKS)))

    def forward(self, xf, xp):
        e = xf @ self.emb                                   # 加权均值池化
        mean = e
        var = xf @ (self.emb ** 2) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-8))        # 加权离散度
        h = self.trunk(torch.cat([mean, std, xp], dim=1))
        return [hd(h) for hd in self.heads]


def train_member(xf_tr, xp_tr, y_tr, m_tr, xf_va, xp_va, y_va, m_va, seed):
    return train_member_v(xf_tr, xp_tr, y_tr, m_tr, xf_va, xp_va, y_va, m_va,
                          seed, bootstrap=False)


def train_member_v(xf_tr, xp_tr, y_tr, m_tr, xf_va, xp_va, y_va, m_va, seed,
                   bootstrap: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiTaskNet(xf_tr.shape[1], xp_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    lossf = nn.MSELoss(reduction="none")

    best, best_state, wait, best_ep = np.inf, None, 0, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        if bootstrap:
            # 每次迭代重采样，让成员之间去相关（否则集成方差严重低估）
            idx = torch.randint(0, len(xf_tr), (len(xf_tr),))
        else:
            idx = torch.arange(len(xf_tr))
        perm = idx[torch.randperm(len(idx))]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 8:
                continue
            outs = model(xf_tr[b], xp_tr[b])
            loss = 0.0
            n_eff = 0.0
            for k in range(len(TASKS)):
                mk = m_tr[b, k]
                if mk.sum() == 0:
                    continue
                r = lossf(outs[k].squeeze(-1), y_tr[b, k]) * mk
                loss = loss + (torch.exp(-model.log_scale[k]) * r
                               + model.log_scale[k]).sum() / mk.sum()
                n_eff += 1
            if n_eff == 0:
                continue
            opt.zero_grad()
            (loss / n_eff).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            ov = model(xf_va, xp_va)
            vloss, cnt = 0.0, 0
            for k in range(len(TASKS)):
                mk = m_va[:, k]
                if mk.sum() == 0:
                    continue
                vloss += lossf(ov[k].squeeze(-1)[mk.bool()],
                               y_va[mk.bool(), k]).mean().item()
                cnt += 1
            vloss = vloss / max(cnt, 1)
        sched.step(vloss)
        if vloss < best - 1e-5:
            best, best_ep, wait = vloss, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best, best_ep


def run_cv(tab, Xf, Xp, Y, M, phys, bootstrap: bool):
    """跑一遍 5 折分组 CV：DL 集成 + 同折 GBM 对照，返回指标与校准结果。"""
    gkf = GroupKFold(5)
    preds = {t: {"dl": np.full(len(tab), np.nan), "gbm": np.full(len(tab), np.nan),
                 "dl_std": np.full(len(tab), np.nan)} for t in TASKS}
    fold_of_row = np.full(len(tab), -1, dtype=int)
    history = []

    for fold, (tr, va) in enumerate(gkf.split(tab, groups=tab["chem_sys"])):
        tr, va = np.asarray(tr), np.asarray(va)
        fold_of_row[va] = fold
        mu = np.nanmean(Y[tr], axis=0)
        sd = np.nanstd(Y[tr], axis=0)
        sd[sd == 0] = 1.0
        Yz = np.nan_to_num((Y - mu) / sd, nan=0.0)

        members, losses = [], []
        for m in range(N_MEMBERS_CV):
            net, bl, be = train_member_v(
                Xf[tr], Xp[tr], torch.tensor(Yz[tr]), torch.tensor(M[tr]),
                Xf[va], Xp[va], torch.tensor(Yz[va]), torch.tensor(M[va]),
                seed=SEED + 100 * fold + m, bootstrap=bootstrap)
            members.append(net)
            losses.append(bl)
            history.append({"fold": fold, "member": m, "val_loss": bl,
                            "best_epoch": be})
        print(f"  fold {fold}: DL 验证损失 " + ", ".join(f"{x:.4f}" for x in losses))

        with torch.no_grad():
            per_member = []
            for net in members:
                o = net(Xf[va], Xp[va])
                per_member.append(
                    torch.cat([oi.detach() for oi in o], dim=1).numpy())
            stack = np.stack(per_member)
        mean_z = stack.mean(axis=0) * sd + mu
        std_z = stack.std(axis=0) * sd

        Xg = phys.to_numpy(dtype=np.float32)
        for k, t in enumerate(TASKS):
            mk = M[tr, k].astype(bool)
            gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                            max_depth=3, min_samples_leaf=2,
                                            random_state=SEED)
            gbm.fit(Xg[tr][mk], Y[tr][mk, k])
            mv = M[va, k].astype(bool)
            preds[t]["gbm"][va[mv]] = gbm.predict(Xg[va][mv])
            preds[t]["dl"][va[mv]] = mean_z[mv, k]
            preds[t]["dl_std"][va[mv]] = std_z[mv, k]

    rows, cal_rows = [], []
    for k, t in enumerate(TASKS):
        mk = ~np.isnan(Y[:, k]) & ~np.isnan(preds[t]["dl"])
        y = Y[mk, k]
        for name in ("dl", "gbm"):
            p = preds[t][name][mk]
            rows.append({"task": t, "unit": TASK_UNITS[t], "model": name,
                         "n": int(mk.sum()), "r2": float(r2_score(y, p)),
                         "mae": float(mean_absolute_error(y, p))})

        resid = np.abs(y - preds[t]["dl"][mk])
        sdd = preds[t]["dl_std"][mk]
        valid = sdd > 1e-9
        ratio = np.full_like(resid, np.inf)
        ratio[valid] = resid[valid] / sdd[valid]
        folds = fold_of_row[mk]
        cov_raw = float(np.mean(resid <= 1.2816 * sdd))
        cov_cal = np.full(len(ratio), np.nan)
        kF = []
        for f in range(5):
            other = (folds != f) & valid
            here = folds == f
            if other.sum() < 20 or here.sum() == 0:
                kF.append(np.nan)
                continue
            kk = float(np.quantile(ratio[other], 0.80))
            kF.append(kk)
            cov_cal[here] = (ratio[here] <= kk).astype(float)
        cov_cal_v = float(np.nanmean(cov_cal))
        cal_rows.append({"task": t, "n": int(mk.sum()),
                         "coverage80_raw": cov_raw,
                         "coverage80_calibrated": cov_cal_v,
                         "k_factor_median": float(np.nanmedian(kF))})
        rows.append({"task": t, "unit": TASK_UNITS[t], "model": "dl_calibration",
                     "n": int(mk.sum()), "r2": np.nan, "mae": np.nan,
                     "coverage80": cov_cal_v})

    per_row = []
    for k, t in enumerate(TASKS):
        mk = ~np.isnan(Y[:, k]) & ~np.isnan(preds[t]["dl"])
        for i in np.where(mk)[0]:
            per_row.append({"task": t, "formula": tab["_red"].iloc[i],
                            "y": float(Y[i, k]), "pred": float(preds[t]["dl"][i]),
                            "std": float(preds[t]["dl_std"][i]),
                            "gbm": float(preds[t]["gbm"][i])})
    return rows, cal_rows, history, per_row


def main() -> int:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    tab = build_multitask_table()
    print(f"多任务表 {len(tab)} 行；每任务标签数 "
          + ", ".join(f"{t}={int(tab[t].notna().sum())}" for t in TASKS))
    print(f"化学体系数 {tab['chem_sys'].nunique()}")

    phys = featlib.build_features(tab["_red"])[PHYS_COLS].astype(float)
    ok = phys.notna().all(axis=1).to_numpy()
    tab, phys = tab[ok].reset_index(drop=True), phys[ok].reset_index(drop=True)
    vocab = element_vocab(tab["_red"])
    frac = frac_matrix(tab["_red"], vocab)
    print(f"可用 {len(tab)} 行；元素词表 {len(vocab)} 个")

    Y = tab[TASKS].to_numpy(dtype=np.float32)
    M = (~np.isnan(Y)).astype(np.float32)
    print("标准化前各任务均值/标准差：",
          {t: (round(float(np.nanmean(tab[t])), 2), round(float(np.nanstd(tab[t])), 2))
           for t in TASKS})

    Xf = torch.tensor(frac)
    Xp = torch.tensor(phys.to_numpy(dtype=np.float32))

    rows, cal_rows, history, per_row = [], [], [], []
    for variant, boot in (("no_bootstrap", False), ("bootstrap", True)):
        print(f"\n===== 变体：{variant} =====")
        r, c, h, pr = run_cv(tab, Xf, Xp, Y, M, phys, boot)
        for d in r + c + h + pr:
            d["variant"] = variant
        rows += r
        cal_rows += c
        history += h
        per_row += pr
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "32_dl_vs_gbm.csv"),
               index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(os.path.join(DATA_DIR, "32_dl_history.csv"),
                                 index=False, encoding="utf-8-sig")
    prow = pd.DataFrame(per_row)
    prow["abs_resid"] = (prow["pred"] - prow["y"]).abs()
    prow.to_csv(os.path.join(DATA_DIR, "32_dl_predictions.csv"),
                index=False, encoding="utf-8-sig")

    print("\n" + "=" * 76)
    print("DL 集成 vs GBM（同折同评估）")
    print("=" * 76)
    print(res.round(4).to_string(index=False))

    print("\n逐任务：DL 相对 GBM 的增量（正=DL 更好）")
    for variant in res["variant"].unique():
        sub = res[res["variant"] == variant]
        for t in TASKS:
            dl = sub[(sub["task"] == t) & (sub["model"] == "dl")]
            gb = sub[(sub["task"] == t) & (sub["model"] == "gbm")]
            if dl.empty or gb.empty:
                continue
            d_r2 = float(dl["r2"].iloc[0]) - float(gb["r2"].iloc[0])
            d_mae = float(gb["mae"].iloc[0]) - float(dl["mae"].iloc[0])
            print(f"  [{variant:12s}] {t:4s}  ΔR²={d_r2:+.4f}  "
                  f"MAE 改善={d_mae:+.3f} {TASK_UNITS[t]}")

    print("\n80% 预测区间覆盖率（目标 0.75~0.85）：")
    for r in cal_rows:
        print(f"  [{r['variant']:12s}] {r['task']:4s} 原始 {r['coverage80_raw']:.3f} -> "
              f"校准后 {r['coverage80_calibrated']:.3f} "
              f"(k≈{r['k_factor_median']:.2f})")

    cal = pd.DataFrame(cal_rows)
    cal.to_csv(os.path.join(DATA_DIR, "32_dl_calibration.csv"),
               index=False, encoding="utf-8-sig")

    print("\n不确定性的区分能力（决定它能否用于后续贝叶斯优化）：")
    print("  spearman = 集成标准差与绝对误差的秩相关；低/高 半区 = 按标准差分半后的 MAE")
    disc = []
    for variant in prow["variant"].unique():
        for t in TASKS:
            g = prow[(prow["variant"] == variant) & (prow["task"] == t)]
            if len(g) < 30:
                continue
            rho = float(g["std"].corr(g["abs_resid"], method="spearman"))
            med = g["std"].median()
            mae_lo = float(g.loc[g["std"] <= med, "abs_resid"].mean())
            mae_hi = float(g.loc[g["std"] > med, "abs_resid"].mean())
            disc.append({"variant": variant, "task": t,
                         "spearman_std_vs_abserr": rho,
                         "mae_low_std_half": mae_lo,
                         "mae_high_std_half": mae_hi,
                         "ratio_high_over_low": mae_hi / mae_lo if mae_lo else np.nan})
            print(f"  [{variant:12s}] {t:4s} spearman={rho:+.3f}  "
                  f"低不确定半区 MAE={mae_lo:.3f} / 高不确定半区 MAE={mae_hi:.3f}  "
                  f"比值={mae_hi / mae_lo if mae_lo else float('nan'):.2f}")
    pd.DataFrame(disc).to_csv(os.path.join(DATA_DIR, "32_dl_uncertainty_value.csv"),
                              index=False, encoding="utf-8-sig")

    idx_path = os.path.join(DATA_DIR, "00_dataset_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, encoding="utf-8-sig", dtype=str).fillna("")
        keys = ["32_dl_vs_gbm.csv", "32_dl_calibration.csv", "32_dl_history.csv",
                "32_dl_predictions.csv", "32_dl_uncertainty_value.csv"]
        keep = idx[~idx["file"].isin(keys)]
        new = pd.DataFrame([
            {"file": "32_dl_vs_gbm.csv", "layer": "建模", "rows": len(res),
             "columns": len(res.columns),
             "source": "scripts/dl_multitask.py（多任务 DL 集成 vs GBM）",
             "purpose": "三任务上 DL 集成与 GBM 的同折对比指标",
             "refresh_note": "重跑 scripts/dl_multitask.py 刷新"},
            {"file": "32_dl_calibration.csv", "layer": "建模", "rows": len(cal),
             "columns": len(cal.columns),
             "source": "scripts/dl_multitask.py",
             "purpose": "深度集成 80% 预测区间的实际覆盖率（不确定性校准）",
             "refresh_note": "重跑 scripts/dl_multitask.py 刷新"},
            {"file": "32_dl_history.csv", "layer": "建模",
             "rows": len(history), "columns": 5,
             "source": "scripts/dl_multitask.py",
             "purpose": "每折每成员的早停轮次与验证损失",
             "refresh_note": "重跑 scripts/dl_multitask.py 刷新"},
            {"file": "32_dl_predictions.csv", "layer": "建模",
             "rows": len(prow), "columns": len(prow.columns),
             "source": "scripts/dl_multitask.py",
             "purpose": "逐样本的折外预测、集成标准差与 GBM 对照预测",
             "refresh_note": "重跑 scripts/dl_multitask.py 刷新"},
            {"file": "32_dl_uncertainty_value.csv", "layer": "建模",
             "rows": len(disc), "columns": 6,
             "source": "scripts/dl_multitask.py",
             "purpose": "不确定性的区分能力：标准差与绝对误差的秩相关、分半 MAE 比",
             "refresh_note": "重跑 scripts/dl_multitask.py 刷新"},
        ])
        pd.concat([keep, new], ignore_index=True).to_csv(
            idx_path, index=False, encoding="utf-8-sig")
        print(f"\n登记 {len(new)} 行 -> 00_dataset_index.csv")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

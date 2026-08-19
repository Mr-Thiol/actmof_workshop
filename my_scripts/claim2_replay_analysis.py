"""
Claim 2 — 离线回放实验
比较两种选点策略在相同查询预算下累计发现的模拟器误差：
  策略 R：随机选点 (Random)
  策略 G：优先选 AND-gate Suspicious 点 (Gate-guided)

关键分层设计：
  - 全量回放：基于所有 95 个实验的累积误差（总误差发现效率）
  - 活跃子集：仅统计 q_true>0 的"真实合成"误差发现（排除平凡零值）
    理由：Suspicious 组中 25/35 实验 q_true=0，模拟器对其误差平凡为 0，
          不需要被发现；我们真正关心的是发现"有意义的模拟器失败"。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── 0. 参数 ──────────────────────────────────────────────────────────────────
DATA_PATH    = r"E:\JupyterPjs\MOF_SynBench\ActMOF_Workshop\Error_data.csv"
TAU_B        = 0.5
N_SEEDS      = 1000
LARGE_ERR_T  = 1.0

# ── 1. 读取并准备数据 ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["B_max"]         = df[["intensity_3000_cross_rate", "fwhm_3_cross_rate"]].max(axis=1)
df["d_min"]         = df["nearest_chebyshev_distance"]
df["abs_log_error"] = df["y_log_abs_error"]
df["is_active"]     = df["q_true"] > 0   # 真实合成成功实验

TAU_D = df["d_min"].median()
df["suspicious"] = (df["B_max"] > TAU_B) & (df["d_min"] > TAU_D)

# 两种误差指标：
#  errors_all   = 所有 95 个实验的 |log error|
#  errors_active = |log error|，但零值实验的"信息量"设为 0（平凡误差不计）
errors_all    = df["abs_log_error"].values
errors_active = np.where(df["is_active"].values, df["abs_log_error"].values, 0.0)
is_susp       = df["suspicious"].values
n_total       = len(df)
n_susp        = is_susp.sum()

print(f"总点数: {n_total}  |  Suspicious: {n_susp}  |  Safe: {n_total - n_susp}")
print(f"门控阈值: tau_B={TAU_B}, tau_d={TAU_D:.1f}")
print(f"Suspicious 组: q_true>0: {(df['suspicious'] & df['is_active']).sum()}  "
      f"| q_true=0: {(df['suspicious'] & ~df['is_active']).sum()}")
print()

# ── 2. 门控策略（确定性，内部随机打乱）──────────────────────────────────────
rng_gate  = np.random.default_rng(42)
susp_idx  = np.where(is_susp)[0]
safe_idx  = np.where(~is_susp)[0]
susp_order = rng_gate.permutation(susp_idx)
safe_order  = rng_gate.permutation(safe_idx)
gate_order  = np.concatenate([susp_order, safe_order])

gate_cum_all    = np.cumsum(errors_all[gate_order])
gate_cum_active = np.cumsum(errors_active[gate_order])
gate_cum_large  = np.cumsum(errors_active[gate_order] > LARGE_ERR_T)

# ── 3. 随机策略（Monte Carlo）───────────────────────────────────────────────
rand_cum_all    = np.zeros((N_SEEDS, n_total))
rand_cum_active = np.zeros((N_SEEDS, n_total))
rand_cum_large  = np.zeros((N_SEEDS, n_total))

rng = np.random.default_rng(0)
for seed in range(N_SEEDS):
    perm = rng.permutation(n_total)
    rand_cum_all[seed]    = np.cumsum(errors_all[perm])
    rand_cum_active[seed] = np.cumsum(errors_active[perm])
    rand_cum_large[seed]  = np.cumsum(errors_active[perm] > LARGE_ERR_T)

def ci95(arr): return 1.96 * arr.std(axis=0) / np.sqrt(N_SEEDS)

r_mean_all    = rand_cum_all.mean(axis=0);    r_ci_all    = ci95(rand_cum_all)
r_mean_active = rand_cum_active.mean(axis=0); r_ci_active = ci95(rand_cum_active)
r_mean_large  = rand_cum_large.mean(axis=0);  r_ci_large  = ci95(rand_cum_large)

# ── 4. 关键统计指标 ──────────────────────────────────────────────────────────
def queries_to_find_k(cum_arr, k):
    idx = np.where(cum_arr >= k)[0]
    return idx[0] + 1 if len(idx) > 0 else np.nan

print("=== 前 K 个有意义大误差（q_true>0 且 |e|>1）的发现速度 ===")
print(f"{'K':>4} | {'Gate':>8} | {'Random(mean)':>14} | {'加速比':>8} | {'Gate 排名':>10}")
for k in [1, 3, 5]:
    g_q  = queries_to_find_k(gate_cum_large, k)
    r_qs = [queries_to_find_k(rand_cum_large[s], k) for s in range(N_SEEDS)]
    r_m  = np.nanmean(r_qs)
    ratio = r_m / g_q if (g_q and not np.isnan(g_q)) else np.nan
    rank  = np.nanmean([rq > g_q for rq in r_qs if not np.isnan(rq)])
    print(f"  {k:>4} | {g_q:>8.0f} | {r_m:>14.1f} | {ratio:>8.2f}x | {rank:>10.1%}")

print()
for label, g_cum, r_cum in [
    ("全量误差 (all)", gate_cum_all, r_mean_all),
    ("活跃误差 (q>0)", gate_cum_active, r_mean_active),
]:
    g_val = g_cum[n_susp - 1]
    r_val = r_cum[n_susp - 1]
    rank  = (rand_cum_active[:, n_susp - 1] < g_val).mean()
    print(f"前 {n_susp} 次查询时 [{label}]:")
    print(f"  Gate={g_val:.2f}  Random(mean)={r_val:.2f}  "
          f"提升={(g_val/r_val-1)*100:.1f}%  Gate优于随机: {rank:.1%}")
print()

# ── 5. 可视化 ─────────────────────────────────────────────────────────────────
steps = np.arange(1, n_total + 1)
c_gate   = "#DD8452"
c_random = "#4C72B0"

fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# (A) 累积活跃误差对比（主图）
ax1 = fig.add_subplot(gs[0, :2])
ax1.fill_between(steps, r_mean_active - r_ci_active, r_mean_active + r_ci_active,
                 alpha=0.25, color=c_random)
ax1.plot(steps, r_mean_active, color=c_random, linewidth=2,
         label=f"Random (mean ± 95% CI, n={N_SEEDS} seeds)")
ax1.plot(steps, gate_cum_active, color=c_gate, linewidth=2.5,
         label="Gate-guided (Suspicious first)")
ax1.axvline(n_susp, color="gray", linestyle="--", linewidth=1.5,
            label=f"Gate pool exhausted (k={n_susp})")
ax1.set_xlabel("Number of queries", fontsize=12)
ax1.set_ylabel("Cumulative active |log error| found\n(q_true > 0 only)", fontsize=11)
ax1.set_title("(A) Meaningful Simulator Errors Discovered\n(Excluding trivial zero-synthesis cases)",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

# (B) 累积大误差数量（活跃 subset）
ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(steps, r_mean_large - r_ci_large, r_mean_large + r_ci_large,
                 alpha=0.25, color=c_random)
ax2.plot(steps, r_mean_large, color=c_random, linewidth=2, label="Random")
ax2.plot(steps, gate_cum_large, color=c_gate, linewidth=2.5, label="Gate-guided")
ax2.axvline(n_susp, color="gray", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Number of queries", fontsize=11)
ax2.set_ylabel(f"Large active errors found\n(q>0 & |e|>{LARGE_ERR_T})", fontsize=11)
ax2.set_title(f"(B) Large-error Discovery\n(meaningful errors only)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

# (C) 累积全量误差（包含零值，用于参考）
ax3 = fig.add_subplot(gs[1, :2])
ax3.fill_between(steps, r_mean_all - r_ci_all, r_mean_all + r_ci_all,
                 alpha=0.25, color=c_random)
ax3.plot(steps, r_mean_all, color=c_random, linewidth=2, label="Random (all errors)")
ax3.plot(steps, gate_cum_all, color=c_gate, linewidth=2.5, label="Gate-guided (all errors)")
ax3.axvline(n_susp, color="gray", linestyle="--", linewidth=1.5, label=f"k={n_susp}")
ax3.set_xlabel("Number of queries", fontsize=12)
ax3.set_ylabel("Cumulative |log error| (all 95 pts)", fontsize=11)
ax3.set_title("(C) Reference: Total Error Discovered (Including Trivial Zeros)\n"
              "[Note: Suspicious pool contains 25 trivial zeros → inflates gap]",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=10); ax3.grid(alpha=0.3)

# (D) Gate 相对 Random 的超额发现量（活跃误差）
ax4 = fig.add_subplot(gs[1, 2])
diff = gate_cum_active - r_mean_active
ax4.fill_between(steps, 0, diff, where=diff > 0, alpha=0.4, color=c_gate,   label="Gate > Random")
ax4.fill_between(steps, 0, diff, where=diff < 0, alpha=0.4, color=c_random, label="Random > Gate")
ax4.plot(steps, diff, color="black", linewidth=1.5)
ax4.axhline(0, color="gray", linewidth=1)
ax4.axvline(n_susp, color="gray", linestyle="--", linewidth=1.5,
            label=f"Suspicious exhausted\n(k={n_susp})")
ax4.set_xlabel("Number of queries", fontsize=11)
ax4.set_ylabel("Δ Cumulative active |log error|\n(Gate − Random mean)", fontsize=10)
ax4.set_title("(D) Gate Advantage Over Random\n(Active subset)", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

fig.suptitle("Claim 2 — Gate-guided Exploration Finds Meaningful Simulator Failures More Efficiently\n"
             f"(Gate Suspicious pool: {(df['suspicious'] & df['is_active']).sum()} active + "
             f"{(df['suspicious'] & ~df['is_active']).sum()} trivial-zero experiments)",
             fontsize=13, fontweight="bold", y=1.01)

OUT_PATH = r"E:\JupyterPjs\MOF_SynBench\ActMOF_Workshop\figures\claim2_replay_analysis.png"
import os; os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n图表已保存: {OUT_PATH}")

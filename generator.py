#!/usr/bin/env python3
"""
Synthetic Trading Strategy Data Generator
==========================================
Reads statistical properties from an input StrategyQuant-style CSV and
generates 1000 synthetic equity curve variations.

Outputs (all inside this script's directory):
  synthetic_strategies.csv  — 1000 strategy parameter rows
  equity_curves.csv         — (n_trades+1) × 1000 equity curve matrix
  summary.png               — visualisation panel
  stats_report.json         — input vs synthetic stats comparison
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_PATH = Path.home() / "Desktop" / "strategyquant_test_data.csv"
OUT_DIR = Path(__file__).parent
N_SYNTHETIC = 1000
N_TRADES = 250
SEED = 42

NUMERIC_COLS = ["Net_Profit", "Drawdown", "Win_Rate", "SQN_Score", "Ret_DD_Ratio"]
CATEGORY_COLORS = {"Good": "#2ecc71", "Medium": "#f39c12", "Bad": "#e74c3c"}

# ── 1. load & analyse ──────────────────────────────────────────────────────────

def load_and_analyse(path: Path):
    df = pd.read_csv(path)
    df["Category"] = df["Category"].str.strip()

    cat_props = df["Category"].value_counts(normalize=True).to_dict()

    # Per-category mean / std / bounds
    stats_by_cat = {}
    for cat, grp in df.groupby("Category"):
        sub = grp[NUMERIC_COLS]
        stats_by_cat[cat] = {
            "mean": sub.mean(),
            "std":  sub.std().fillna(sub.mean() * 0.10),
            "min":  sub.min(),
            "max":  sub.max(),
        }

    # Global covariance (for correlated sampling fallback)
    cov = df[NUMERIC_COLS].cov()

    return df, stats_by_cat, cov, cat_props


# ── 2. correlated parameter sampling ──────────────────────────────────────────

def _truncnorm_sample(rng, mean, std, lo, hi, size=1):
    """Draw from a truncated normal clipped to [lo, hi]."""
    if std <= 0:
        return np.full(size, mean)
    a, b = (lo - mean) / std, (hi - mean) / std
    result = truncnorm.rvs(a, b, loc=mean, scale=std, size=size,
                           random_state=int(rng.integers(0, 2**31)))
    return result


def generate_params(stats_by_cat: dict, cat_props: dict, n: int = 1000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    cats = list(cat_props.keys())
    weights = [cat_props[c] for c in cats]

    rows = []
    for i in range(n):
        cat = rng.choice(cats, p=weights)
        s = stats_by_cat[cat]

        # Widen bounds slightly to allow variation beyond original extremes
        def samp(col, lo_mult=0.70, hi_mult=1.30):
            val = _truncnorm_sample(
                rng,
                mean=float(s["mean"][col]),
                std=float(s["std"][col]),
                lo=float(s["min"][col]) * lo_mult,
                hi=float(s["max"][col]) * hi_mult,
            )
            return float(val.flat[0]) if hasattr(val, "flat") else float(val)

        win_rate  = round(np.clip(samp("Win_Rate"),  20.0, 80.0), 2)
        sqn       = round(np.clip(samp("SQN_Score"), 0.5,  5.0),  2)
        drawdown  = round(np.clip(samp("Drawdown"),  500,  8000),  2)
        ret_dd    = round(np.clip(samp("Ret_DD_Ratio"), 1.0, 8.0), 2)
        net_profit = round(ret_dd * drawdown, 2)

        rows.append({
            "Strategy_ID":   f"SYN_{i + 1:04d}",
            "Net_Profit":    net_profit,
            "Drawdown":      drawdown,
            "Win_Rate":      win_rate,
            "SQN_Score":     sqn,
            "Ret_DD_Ratio":  ret_dd,
            "Category":      cat,
        })

    return pd.DataFrame(rows)


# ── 3. equity curve simulation ─────────────────────────────────────────────────

def simulate_equity_curve(
    net_profit: float,
    drawdown: float,
    win_rate: float,
    n_trades: int = N_TRADES,
    seed: int | None = None,
) -> np.ndarray:
    """
    Trade-by-trade Monte Carlo equity curve.

    Win sizes ~ Exponential(avg_win), loss sizes ~ Exponential(avg_loss).
    avg_win / avg_loss (profit factor) is derived from net_profit and drawdown
    so the simulated curve broadly hits both targets.
    """
    rng = np.random.default_rng(seed)

    wr = win_rate / 100.0
    lossrate = 1.0 - wr

    # Profit factor: solve so that expectation per trade ≈ net_profit / n_trades
    # E[PnL/trade] = wr*avg_win - lossrate*avg_loss = net_profit / n_trades
    # We set avg_loss proportional to drawdown and derive avg_win from PF.
    avg_loss = max(drawdown / (n_trades * lossrate * 0.40 + 1e-6), 20.0)
    target_exp = net_profit / n_trades
    # wr*avg_win = target_exp + lossrate*avg_loss
    avg_win = max((target_exp + lossrate * avg_loss) / max(wr, 1e-6), avg_loss * 1.05)

    wins  = rng.exponential(avg_win,  n_trades)
    losses = rng.exponential(avg_loss, n_trades)
    outcomes = rng.random(n_trades) < wr
    trade_pnl = np.where(outcomes, wins, -losses)

    # Add mild autocorrelation / momentum effect (regime-like clustering)
    momentum = rng.normal(0, avg_loss * 0.05, n_trades)
    trade_pnl += np.convolve(momentum, np.ones(5) / 5, mode="same")

    # Scale so final equity matches net_profit exactly
    total = trade_pnl.sum()
    if abs(total) > 1e-6:
        trade_pnl *= net_profit / total

    equity = np.concatenate([[0.0], np.cumsum(trade_pnl)])
    return equity


def build_all_curves(df: pd.DataFrame) -> dict[str, np.ndarray]:
    curves = {}
    for _, row in df.iterrows():
        sid = row["Strategy_ID"]
        seed_val = int(sid.split("_")[1])
        curves[sid] = simulate_equity_curve(
            net_profit=row["Net_Profit"],
            drawdown=row["Drawdown"],
            win_rate=row["Win_Rate"],
            seed=seed_val,
        )
    return curves


# ── 4. save outputs ────────────────────────────────────────────────────────────

def save_equity_curves(curves: dict[str, np.ndarray], outdir: Path) -> Path:
    max_len = max(len(v) for v in curves.values())
    data = {}
    for sid, curve in curves.items():
        col = np.full(max_len, np.nan)
        col[: len(curve)] = curve
        data[sid] = col
    df = pd.DataFrame(data)
    df.index.name = "Trade"
    path = outdir / "equity_curves.csv"
    df.to_csv(path)
    return path


def save_stats_report(df_orig, df_synth, outdir: Path) -> Path:
    def describe(df):
        return {
            "n": len(df),
            "category_proportions": df["Category"].value_counts(normalize=True).round(3).to_dict(),
            "stats": df[NUMERIC_COLS].describe().round(2).to_dict(),
        }

    report = {"input": describe(df_orig), "synthetic": describe(df_synth)}
    path = outdir / "stats_report.json"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    return path


# ── 5. visualisation ───────────────────────────────────────────────────────────

_DARK_BG   = "#0d1117"
_PANEL_BG  = "#161b22"
_BORDER    = "#30363d"
_TEXT_DIM  = "#8b949e"
_TEXT_MAIN = "#c9d1d9"
_TEXT_BRIGHT = "#e6edf3"


def _style_ax(ax):
    ax.set_facecolor(_PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(_BORDER)
    ax.tick_params(colors=_TEXT_DIM, labelsize=8)
    ax.xaxis.label.set_color(_TEXT_MAIN)
    ax.yaxis.label.set_color(_TEXT_MAIN)
    ax.title.set_color(_TEXT_BRIGHT)


def plot_summary(df: pd.DataFrame, curves: dict[str, np.ndarray], outdir: Path) -> Path:
    fig = plt.figure(figsize=(20, 15), facecolor=_DARK_BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_equity  = fig.add_subplot(gs[0, :])
    ax_profit  = fig.add_subplot(gs[1, 0])
    ax_dd      = fig.add_subplot(gs[1, 1])
    ax_wr      = fig.add_subplot(gs[1, 2])
    ax_sqn     = fig.add_subplot(gs[2, 0])
    ax_scatter = fig.add_subplot(gs[2, 1])
    ax_cat     = fig.add_subplot(gs[2, 2])

    for ax in [ax_equity, ax_profit, ax_dd, ax_wr, ax_sqn, ax_scatter, ax_cat]:
        _style_ax(ax)

    # ── equity curves (up to 60 per category for legibility) ──
    ax_equity.set_title("Synthetic Equity Curves — 1000 Variations (sample shown)",
                        fontsize=11, fontweight="bold")
    for cat, color in CATEGORY_COLORS.items():
        ids = df[df["Category"] == cat]["Strategy_ID"].tolist()
        for sid in ids[:60]:
            ax_equity.plot(curves[sid], color=color, alpha=0.12, linewidth=0.55)
    for cat, color in CATEGORY_COLORS.items():
        ax_equity.plot([], [], color=color, label=cat, linewidth=1.8)
    ax_equity.axhline(0, color=_BORDER, linewidth=0.8, linestyle="--")
    ax_equity.legend(fontsize=9, facecolor=_PANEL_BG, edgecolor=_BORDER, labelcolor=_TEXT_MAIN)
    ax_equity.set_xlabel("Trade #")
    ax_equity.set_ylabel("Cumulative P&L")

    # ── histograms ──
    hist_cfg = [
        (ax_profit, "Net_Profit",   "Net Profit"),
        (ax_dd,     "Drawdown",     "Drawdown"),
        (ax_wr,     "Win_Rate",     "Win Rate (%)"),
        (ax_sqn,    "SQN_Score",    "SQN Score"),
    ]
    for ax, col, title in hist_cfg:
        for cat, color in CATEGORY_COLORS.items():
            vals = df[df["Category"] == cat][col]
            ax.hist(vals, bins=25, color=color, alpha=0.60, edgecolor="none", label=cat)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Count")

    # ── scatter: SQN vs Net Profit ──
    for cat, color in CATEGORY_COLORS.items():
        sub = df[df["Category"] == cat]
        ax_scatter.scatter(sub["SQN_Score"], sub["Net_Profit"],
                           c=color, s=14, alpha=0.55, edgecolors="none", label=cat)
    ax_scatter.set_title("Net Profit vs SQN Score", fontsize=9, fontweight="bold")
    ax_scatter.set_xlabel("SQN Score")
    ax_scatter.set_ylabel("Net Profit")
    ax_scatter.legend(fontsize=7, facecolor=_PANEL_BG, edgecolor=_BORDER, labelcolor=_TEXT_MAIN)

    # ── category bar ──
    cat_counts = df["Category"].value_counts().reindex(CATEGORY_COLORS.keys(), fill_value=0)
    bar_colors = [CATEGORY_COLORS[c] for c in cat_counts.index]
    bars = ax_cat.bar(cat_counts.index, cat_counts.values, color=bar_colors, edgecolor="none")
    for bar, val in zip(bars, cat_counts.values):
        ax_cat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(val), ha="center", va="bottom", color=_TEXT_MAIN, fontsize=8)
    ax_cat.set_title("Category Distribution", fontsize=9, fontweight="bold")
    ax_cat.set_ylabel("Count")

    fig.suptitle(
        "Synthetic Algorithmic Trading Strategy Lab  ·  1000 Equity Curve Variations",
        fontsize=13, fontweight="bold", color=_TEXT_BRIGHT, y=0.99,
    )

    path = outdir / "summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
    plt.close(fig)
    return path


# ── 6. main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Input  : {INPUT_PATH}")
    print(f"Output : {OUT_DIR}\n")

    print("[1/6] Loading and analysing input data …")
    df_orig, stats_by_cat, cov, cat_props = load_and_analyse(INPUT_PATH)
    print(f"       {len(df_orig)} strategies | categories: {list(cat_props)}")
    print(f"       Column stats:\n{df_orig[NUMERIC_COLS].describe().round(1).to_string()}\n")

    print("[2/6] Sampling 1000 synthetic strategy parameters …")
    df_synth = generate_params(stats_by_cat, cat_props, n=N_SYNTHETIC, seed=SEED)
    synth_csv = OUT_DIR / "synthetic_strategies.csv"
    df_synth.to_csv(synth_csv, index=False)
    print(f"       Saved → {synth_csv.name}")

    print("[3/6] Simulating equity curves …")
    curves = build_all_curves(df_synth)
    print(f"       {len(curves)} curves × {N_TRADES + 1} points each")

    print("[4/6] Saving equity_curves.csv …")
    ep = save_equity_curves(curves, OUT_DIR)
    print(f"       Saved → {ep.name}  ({ep.stat().st_size / 1e6:.1f} MB)")

    print("[5/6] Rendering summary.png …")
    cp = plot_summary(df_synth, curves, OUT_DIR)
    print(f"       Saved → {cp.name}")

    print("[6/6] Writing stats_report.json …")
    rp = save_stats_report(df_orig, df_synth, OUT_DIR)
    print(f"       Saved → {rp.name}")

    print("\n── Summary ───────────────────────────────────────────────")
    print(df_synth["Category"].value_counts().to_string())
    print()
    print(df_synth[NUMERIC_COLS].describe().round(2).to_string())
    print("\nAll done.")


if __name__ == "__main__":
    main()

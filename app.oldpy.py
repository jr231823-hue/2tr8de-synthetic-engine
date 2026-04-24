#!/usr/bin/env python3
"""2TR8DE Synthetic Lab — Flask Backend
Handles three real SQX/broker export formats automatically:
  1. SQX trade log   — semicolon-delimited, quoted (Strategy X.Y.Z.csv)
  2. Live statement  — comma-delimited, Profit + Magic Number columns
  3. Summary table   — one row per strategy with aggregated metrics
"""

import io
import json
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from scipy.stats import truncnorm

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB for bulk files

NUMERIC_COLS = ["Net_Profit", "Drawdown", "Win_Rate", "SQN_Score", "Ret_DD_Ratio"]
N_SYNTHETIC  = 1000
SEED         = 42

# ── column alias maps ─────────────────────────────────────────────────────────
# Keys are the standard internal names; values are lists of known SQX/broker aliases.
SUMMARY_ALIASES = {
    "Strategy_ID":  ["strategy_id", "id", "name", "strategy", "strategy name",
                     "strat", "strat_id", "strategy id", "algo", "system"],
    "Net_Profit":   ["net_profit", "netprofit", "net profit", "total profit",
                     "totalprofit", "profit", "total p&l", "total pnl"],
    "Drawdown":     ["drawdown", "max_drawdown", "maxdrawdown", "max drawdown",
                     "maximum drawdown", "max dd"],
    "Win_Rate":     ["win_rate", "winrate", "win rate", "win%", "win %",
                     "percent profitable", "% profitable", "pct_win"],
    "SQN_Score":    ["sqn_score", "sqn", "sqn score", "system quality number",
                     "system quality", "sqn2"],
    "Ret_DD_Ratio": ["ret_dd_ratio", "ret/dd", "ret_dd", "return/drawdown",
                     "return / drawdown", "profit factor", "profitfactor",
                     "return on drawdown", "r/dd"],
    "Category":     ["category", "rating", "grade", "quality", "rank"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FORMAT DETECTION & PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip quotes and whitespace from all column names and string values."""
    df = df.copy()
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip('"').str.strip("'")
    return df


def detect_and_parse(file_bytes: bytes) -> tuple[str, pd.DataFrame, dict]:
    """
    Auto-detect format and return (format_name, cleaned_df, info_dict).

    format_name is one of:
      'sqx_trade_log'  — real SQX strategy export (semicolon, Profit/Loss column)
      'statement'      — live broker/analytics statement (comma, Profit + Action)
      'summary'        — aggregated strategy table (Net_Profit or aliases, Category)
    """
    text = file_bytes.decode("utf-8", errors="replace")
    first_line = text.split("\n")[0]

    # Detect delimiter from the header row
    semi_count  = first_line.count(";")
    comma_count = first_line.count(",")
    sep = ";" if semi_count > comma_count else ","

    df = pd.read_csv(io.StringIO(text), sep=sep, quotechar='"',
                     on_bad_lines="skip", dtype=str)
    df = _clean_df(df)
    cols_lower = {c.lower() for c in df.columns}

    # ── SQX trade log ──────────────────────────────────────────────────────────
    # Definitive markers: Profit/Loss AND (Sample type OR Balance)
    if "profit/loss" in cols_lower and (
        "sample type" in cols_lower or "balance" in cols_lower
    ):
        return "sqx_trade_log", df, {"sep": sep}

    # ── Live statement ─────────────────────────────────────────────────────────
    # Has a profit column AND trade-identifying columns (action or magic number)
    if "profit" in cols_lower and (
        "action" in cols_lower or "magic number" in cols_lower
    ):
        return "statement", df, {"sep": sep}

    # ── Summary table ─────────────────────────────────────────────────────────
    # Has at least one recognised profit alias and one recognised category alias
    profit_aliases  = {a for a in SUMMARY_ALIASES["Net_Profit"]}
    category_aliases = {a for a in SUMMARY_ALIASES["Category"]}
    sqn_aliases      = {a for a in SUMMARY_ALIASES["SQN_Score"]}

    has_profit   = bool(cols_lower & profit_aliases)
    has_category = bool(cols_lower & category_aliases)
    has_sqn      = bool(cols_lower & sqn_aliases)

    if has_profit and (has_category or has_sqn):
        return "summary", df, {"sep": sep}

    # ── Unknown ────────────────────────────────────────────────────────────────
    return "unknown", df, {"columns": list(df.columns), "sep": sep}


# ── SQX multi-strategy split (repeated-header detection) ─────────────────────

def split_concatenated_sqx(text: str) -> list[tuple[str, str]]:
    """
    Detect and split a file that concatenates multiple SQX trade logs.

    Handles two common forms:
      A) Header repeats directly:
           Ticket;Symbol;...   ← header
           1;XAUUSD;...
           Ticket;Symbol;...   ← header again
           1;XAUUSD;...

      B) Strategy name line precedes each header:
           Strategy 1.10.17
           Ticket;Symbol;...
           1;XAUUSD;...
           Strategy 3.7.25
           Ticket;Symbol;...
           1;XAUUSD;...

    Returns a list of (strategy_name, csv_text) pairs, or [] if only one block.
    """
    lines = text.splitlines()
    if not lines:
        return []

    # Find the canonical CSV header — the first line that looks like a SQX header
    # (contains a known column name and a field delimiter).
    header: str | None = None
    for line in lines:
        stripped = line.strip()
        lo = stripped.lower()
        if (";" in stripped or "," in stripped) and (
            "profit/loss" in lo or "profit" in lo or "ticket" in lo
        ):
            header = stripped
            break

    if header is None:
        return []

    # Locate every occurrence of the header in the file
    split_at: list[int] = []
    for i, line in enumerate(lines):
        if line.strip() == header:
            split_at.append(i)

    if len(split_at) < 2:
        return []

    segments: list[tuple[str, str]] = []
    for idx, start in enumerate(split_at):
        end = split_at[idx + 1] if idx + 1 < len(split_at) else len(lines)

        # Look backward from the header for a strategy name line (skip blank lines;
        # accept a line only if it has no field delimiter — i.e. plain text).
        name: str | None = None
        check = start - 1
        while check >= 0 and not lines[check].strip():
            check -= 1
        if check >= 0:
            candidate = lines[check].strip().strip('"').strip("'")
            if candidate and ";" not in candidate and "," not in candidate:
                name = candidate
        if not name:
            name = f"Strategy_{idx + 1}"

        block = "\n".join(lines[start:end])
        segments.append((name, block))

    return segments


def group_by_comment(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    If the 'Comment' column carries distinct non-empty strategy names, group
    trade P&L by those names and return a dict of name → pnl array.
    Returns {} if the column doesn't exist or has only one distinct value.
    """
    col_map = {c.lower(): c for c in df.columns}
    comment_col = col_map.get("comment")
    pnl_col     = col_map.get("profit/loss")
    if comment_col is None or pnl_col is None:
        return {}

    comments = df[comment_col].astype(str).str.strip()
    pnl      = pd.to_numeric(df[pnl_col], errors="coerce")

    # Discard empty / numeric-looking comments
    meaningful = comments[comments.str.len() > 0]
    meaningful = meaningful[~meaningful.str.match(r"^\d+\.?\d*$")]
    unique_names = [n for n in meaningful.unique() if n not in ("nan", "", "0")]

    if len(unique_names) < 2:
        return {}

    groups: dict[str, np.ndarray] = {}
    for name in unique_names:
        mask = comments == name
        vals = pnl[mask].dropna().to_numpy(dtype=float)
        if len(vals) >= 10:
            groups[name] = vals
    return groups


# ── SQX trade log ─────────────────────────────────────────────────────────────

def parse_sqx_trade_log(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series | None]:
    """
    Extract (trade_pnl, balance, sample_type) from a SQX trade-log DataFrame.
    Returns numpy arrays ready for metric computation.
    """
    col_map = {c.lower(): c for c in df.columns}

    pnl_col     = col_map.get("profit/loss")
    balance_col = col_map.get("balance")
    sample_col  = col_map.get("sample type")
    comm_col    = col_map.get("comm/swap")

    if pnl_col is None:
        raise ValueError("Could not find 'Profit/Loss' column in SQX trade log.")

    pnl = pd.to_numeric(df[pnl_col], errors="coerce")

    # Include commissions/swaps in the net P&L if present
    if comm_col:
        comm = pd.to_numeric(df[comm_col], errors="coerce").fillna(0)
        pnl = pnl + comm

    balance      = pd.to_numeric(df[balance_col], errors="coerce") if balance_col else None
    sample_type  = df[sample_col] if sample_col else None

    # Drop rows where P&L is NaN (header repeats / blank lines)
    valid = pnl.notna()
    pnl         = pnl[valid].reset_index(drop=True)
    balance     = balance[valid].reset_index(drop=True) if balance is not None else None
    sample_type = sample_type[valid].reset_index(drop=True) if sample_type is not None else None

    return pnl.to_numpy(dtype=float), \
           (balance.to_numpy(dtype=float) if balance is not None else None), \
           sample_type


# ── Live statement ─────────────────────────────────────────────────────────────

def parse_statement(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Parse a live broker statement into a dict of strategy_name → trade_pnl array.
    Strategy names come from the 'Comment' or 'Magic Number' column.
    Falls back to treating all trades as one portfolio if no strategy column exists.
    """
    col_map = {c.lower(): c for c in df.columns}
    pnl_col = col_map.get("profit")
    if pnl_col is None:
        raise ValueError("No 'Profit' column found in statement.")

    pnl = pd.to_numeric(df[pnl_col], errors="coerce")

    # Add commission + swap if present
    for aux in ["commission", "swap"]:
        if aux in col_map:
            aux_vals = pd.to_numeric(df[col_map[aux]], errors="coerce").fillna(0)
            pnl = pnl + aux_vals

    # Identify strategy grouping column
    strat_col = col_map.get("comment") or col_map.get("magic number")
    if strat_col:
        df = df.copy()
        df["_pnl"] = pnl
        grouped = {}
        for name, grp in df.groupby(strat_col):
            vals = grp["_pnl"].dropna().to_numpy(dtype=float)
            if len(vals) >= 10:
                grouped[str(name)] = vals
        if grouped:
            return grouped

    # No usable grouping — treat as single strategy
    valid = pnl.notna().to_numpy()
    return {"Portfolio": pnl[valid].to_numpy(dtype=float)}


# ── Summary table ──────────────────────────────────────────────────────────────

def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliased columns to the standard internal names."""
    col_lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for std_name, aliases in SUMMARY_ALIASES.items():
        if std_name in df.columns:
            continue  # already correct
        for alias in aliases:
            if alias in col_lower_map:
                rename[col_lower_map[alias]] = std_name
                break
    df = df.rename(columns=rename)

    # If Category is missing, derive it from SQN_Score
    if "Category" not in df.columns and "SQN_Score" in df.columns:
        sqn = pd.to_numeric(df["SQN_Score"], errors="coerce")
        df["Category"] = sqn.apply(sqn_to_category)

    # Ensure numeric columns are numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=[c for c in NUMERIC_COLS if c in df.columns])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. METRIC COMPUTATION FROM RAW TRADES
# ═══════════════════════════════════════════════════════════════════════════════

def sqn_to_category(sqn: float) -> str:
    if sqn >= 2.5:
        return "Good"
    elif sqn >= 2.0:
        return "Medium"
    return "Bad"


def max_drawdown_dollars(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown in absolute dollar terms."""
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def compute_sqn(trade_pnl: np.ndarray) -> float:
    """
    SQN = sqrt(min(N, 100)) * mean(R) / std(R)
    Capping at N=100 prevents inflation from large trade counts and
    matches typical StrategyQuant display ranges (1–5).
    """
    n = len(trade_pnl)
    if n < 5:
        return 0.0
    mean_r = np.mean(trade_pnl)
    std_r  = np.std(trade_pnl)
    if std_r < 1e-9:
        return 0.0
    return float(np.sqrt(min(n, 100)) * mean_r / std_r)


def trades_to_metrics(trade_pnl: np.ndarray, strategy_id: str = "INPUT") -> dict:
    """Compute all five standard metrics from a trade P&L array."""
    equity      = np.concatenate([[0.0], np.cumsum(trade_pnl)])
    net_profit  = float(equity[-1])
    drawdown    = max(max_drawdown_dollars(equity), 1.0)
    win_rate    = float(np.mean(trade_pnl > 0) * 100)
    sqn         = compute_sqn(trade_pnl)
    ret_dd      = net_profit / drawdown
    category    = sqn_to_category(sqn)

    return {
        "Strategy_ID":  strategy_id,
        "Net_Profit":   round(net_profit, 2),
        "Drawdown":     round(drawdown, 2),
        "Win_Rate":     round(win_rate, 2),
        "SQN_Score":    round(sqn, 2),
        "Ret_DD_Ratio": round(ret_dd, 2),
        "Category":     category,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SYNTHETIC GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_single_strategy(
    trade_pnl: np.ndarray,
    n: int = N_SYNTHETIC,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate n synthetic equity curves by bootstrapping the real trade sequence.
    Each iteration resamples trades with replacement and adds mild noise,
    preserving the original trade-return distribution while testing path sensitivity.
    """
    rng   = np.random.default_rng(seed)
    n_trades = len(trade_pnl)
    std_pnl  = np.std(trade_pnl)
    rows     = []
    curves   = {}

    for i in range(n):
        sid = f"SYN_{i+1:04d}"

        # Resample with replacement
        idx = rng.integers(0, n_trades, size=n_trades)
        sample = trade_pnl[idx].copy()

        # Add small Gaussian noise (3 % of trade std) to avoid pure repetition
        sample += rng.normal(0, std_pnl * 0.03, n_trades)

        equity = np.concatenate([[0.0], np.cumsum(sample)])
        curves[sid] = equity

        m = trades_to_metrics(sample, sid)
        rows.append(m)

    return pd.DataFrame(rows), curves


def _truncnorm_sample(rng, mean, std, lo, hi):
    if std <= 0:
        return mean
    a, b = (lo - mean) / std, (hi - mean) / std
    result = truncnorm.rvs(a, b, loc=mean, scale=std, size=1,
                           random_state=int(rng.integers(0, 2**31)))
    return float(result.flat[0])


def generate_params_from_summary(stats_by_cat, cat_props, n=N_SYNTHETIC, seed=SEED):
    """Generate synthetic strategy parameter rows from a summary-table distribution."""
    rng  = np.random.default_rng(seed)
    cats = list(cat_props.keys())
    weights = [cat_props[c] for c in cats]
    rows = []

    for i in range(n):
        cat = rng.choice(cats, p=weights)
        s   = stats_by_cat[cat]

        def samp(col, lo_m=0.70, hi_m=1.30):
            return _truncnorm_sample(
                rng,
                mean=float(s["mean"][col]),
                std=float(s["std"][col]),
                lo=float(s["min"][col]) * lo_m,
                hi=float(s["max"][col]) * hi_m,
            )

        drawdown = np.clip(samp("Drawdown"), 500, 8000)
        ret_dd   = np.clip(samp("Ret_DD_Ratio"), 1.0, 8.0)
        rows.append({
            "Strategy_ID":  f"SYN_{i+1:04d}",
            "Net_Profit":   round(float(ret_dd * drawdown), 2),
            "Drawdown":     round(float(drawdown), 2),
            "Win_Rate":     round(np.clip(samp("Win_Rate"), 20.0, 80.0), 2),
            "SQN_Score":    round(np.clip(samp("SQN_Score"), 0.5, 5.0), 2),
            "Ret_DD_Ratio": round(float(ret_dd), 2),
            "Category":     cat,
        })

    return pd.DataFrame(rows)


def simulate_curve_from_params(net_profit, drawdown, win_rate, n_trades=250, seed=None):
    """Parametric equity curve used for summary-format inputs."""
    rng      = np.random.default_rng(seed)
    wr       = win_rate / 100.0
    lossrate = 1.0 - wr
    avg_loss = max(drawdown / (n_trades * lossrate * 0.40 + 1e-6), 20.0)
    target   = net_profit / n_trades
    avg_win  = max((target + lossrate * avg_loss) / max(wr, 1e-6), avg_loss * 1.05)
    wins     = rng.exponential(avg_win, n_trades)
    losses   = rng.exponential(avg_loss, n_trades)
    pnl      = np.where(rng.random(n_trades) < wr, wins, -losses)
    momentum = rng.normal(0, avg_loss * 0.05, n_trades)
    pnl     += np.convolve(momentum, np.ones(5) / 5, mode="same")
    total    = pnl.sum()
    if abs(total) > 1e-6:
        pnl *= net_profit / total
    return np.concatenate([[0.0], np.cumsum(pnl)])


def build_curves_from_summary(df):
    return {
        row["Strategy_ID"]: simulate_curve_from_params(
            row["Net_Profit"], row["Drawdown"], row["Win_Rate"],
            seed=int(row["Strategy_ID"].split("_")[1])
        )
        for _, row in df.iterrows()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BULK SCORING  (direct — no Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════════

def score_single_strategy(row) -> tuple[float, str]:
    """
    Score one strategy row directly from its five metrics.
    Returns (composite_0_to_100, grade_letter).
    No simulation — runs in microseconds, safe for thousands of rows.
    """
    sqn        = float(row.get("SQN_Score",    0) or 0)
    ret_dd     = float(row.get("Ret_DD_Ratio", 0) or 0)
    win_rate   = float(row.get("Win_Rate",     0) or 0)
    net_profit = float(row.get("Net_Profit",   0) or 0)
    category   = str(row.get("Category", "Bad"))

    sqn_s    = min(100.0, max(0.0, (sqn      - 1.0) / (3.5 - 1.0) * 100))
    risk_s   = min(100.0, max(0.0, (ret_dd   - 1.5) / (5.0 - 1.5) * 100))
    wr_s     = min(100.0, max(0.0, (win_rate - 35)  / (65  - 35)  * 100))
    cat_s    = {"Good": 100.0, "Medium": 55.0, "Bad": 10.0}.get(category, 10.0)
    profit_s = 100.0 if net_profit > 0 else 0.0

    composite = (0.35 * sqn_s + 0.25 * risk_s + 0.15 * wr_s
                 + 0.15 * cat_s + 0.10 * profit_s)
    composite = round(composite, 1)

    grade = ("A" if composite >= 85 else
             "B" if composite >= 70 else
             "C" if composite >= 55 else
             "D" if composite >= 40 else "F")
    return composite, grade


def build_bulk_results(df: pd.DataFrame, trade_map: dict[str, np.ndarray] | None = None, top_mc: int = 5) -> tuple[list[dict], dict]:
    """
    Score every row in df, return (ranked_strategies, summary_counts).
    Preserves all original columns so the CSV export is faithful to the source.

    PRO upgrade:
    - First pass: fast direct score for all strategies.
    - Second pass: the top N candidates get Monte Carlo Pro scoring.
      If raw trades are available, it uses the real trade sequence.
      If raw trades are not available, it reconstructs a synthetic trade path
      from summary metrics and stress-tests that path.
    """
    trade_map = trade_map or {}
    # Collect all columns that exist beyond the five standard ones
    extra_cols = [c for c in df.columns
                  if c not in NUMERIC_COLS + ["Strategy_ID", "Category"]]

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        composite, grade = score_single_strategy(row)

        sid = str(row.get("Strategy_ID", "")).strip()
        if not sid or sid == "nan":
            sid = f"STRAT_{i + 1:04d}"

        entry = {
            "rank":       0,          # filled after sort
            "id":         sid,
            "score":      composite,
            "grade":      grade,
            "net_profit": round(float(row.get("Net_Profit",   0) or 0), 2),
            "drawdown":   round(float(row.get("Drawdown",     0) or 0), 2),
            "win_rate":   round(float(row.get("Win_Rate",     0) or 0), 2),
            "sqn_score":  round(float(row.get("SQN_Score",    0) or 0), 2),
            "ret_dd":     round(float(row.get("Ret_DD_Ratio", 0) or 0), 2),
            "category":   str(row.get("Category", "Bad")),
        }
        # carry extra source columns through for the export
        for col in extra_cols:
            entry[f"_x_{col}"] = str(row.get(col, ""))
        rows.append(entry)

    # ── First pass ranking using fast score ─────────────────────────────────
    rows.sort(key=lambda x: x["score"], reverse=True)

    # ── Second pass: Monte Carlo Pro for top candidates ─────────────────────
    # This keeps the dashboard fast while giving serious analysis to the
    # strategies that are actually candidates.
    for i, r in enumerate(rows[:max(0, int(top_mc))]):
        try:
            sid = str(r["id"])

            # Prefer real trade sequence when available.
            if sid in trade_map and len(trade_map[sid]) >= 10:
                trade_pnl = np.asarray(trade_map[sid], dtype=float)
            else:
                # Fallback for summary-only rows: reconstruct a trade path from
                # the metrics, then stress-test it. This is not as strong as raw
                # trades, but it is much better than no Monte Carlo at all.
                curve = simulate_curve_from_params(
                    r["net_profit"],
                    max(r["drawdown"], 1.0),
                    r["win_rate"],
                    n_trades=250,
                    seed=SEED + i,
                )
                trade_pnl = np.diff(curve)

            df_synth, curves = bootstrap_single_strategy(
                trade_pnl,
                n=min(N_SYNTHETIC, 500),
                seed=SEED + i,
            )
            pro_score = compute_robustness(df_synth, curves)

            r["score"] = pro_score["composite"]
            r["grade"] = pro_score["grade"]
            r["recommendation"] = pro_score["recommendation"]
            r["explanation"] = pro_score["explanation"]
            r["mc_pro"] = True
            r["mc_stats"] = pro_score.get("stats", {})
        except Exception as e:
            r["mc_pro"] = False
            r["mc_error"] = str(e)

    # Re-sort after Monte Carlo Pro because the real stress test can change rank.
    rows.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r.setdefault("recommendation", _recommendation_from_score(r["score"]))
        r.setdefault("explanation", "Fast score only. Monte Carlo Pro was reserved for the top candidates.")
        r.setdefault("mc_pro", False)

    deploy   = sum(1 for r in rows if r["grade"] in ("A", "B"))
    optimize = sum(1 for r in rows if r["grade"] == "C")
    discard  = sum(1 for r in rows if r["grade"] in ("D", "F"))

    summary = {
        "total":    len(rows),
        "deploy":   deploy,
        "optimize": optimize,
        "discard":  discard,
        "extra_cols": extra_cols,
    }
    return rows, summary


def build_bulk_score(df: pd.DataFrame, strategies: list[dict]) -> dict:
    """Aggregate score for the whole portfolio (mean of individual scores)."""
    scores = [s["score"] for s in strategies]
    composite = round(float(np.mean(scores)), 1)
    grade = ("A" if composite >= 85 else
             "B" if composite >= 70 else
             "C" if composite >= 55 else
             "D" if composite >= 40 else "F")

    avg_sqn      = float(df["SQN_Score"].mean())
    positive_pct = float((df["Net_Profit"] > 0).mean() * 100)
    med_ret_dd   = float(df["Ret_DD_Ratio"].median())
    good_pct     = float((df["Category"] == "Good").mean() * 100)

    sqn_s  = min(100, max(0, (avg_sqn   - 1.0) / (3.5 - 1.0) * 100))
    risk_s = min(100, max(0, (med_ret_dd - 1.5) / (5.0 - 1.5) * 100))

    return {
        "composite": composite,
        "grade": grade,
        "components": {
            "sqn_quality":       round(sqn_s, 1),
            "profit_stability":  round(positive_pct, 1),
            "risk_efficiency":   round(risk_s, 1),
            "category_health":   round(good_pct, 1),
            "curve_consistency": 0.0,
        },
        "stats": {
            "avg_sqn":       round(avg_sqn, 2),
            "positive_pct":  round(positive_pct, 1),
            "median_ret_dd": round(med_ret_dd, 2),
            "good_pct":      round(good_pct, 1),
        },
    }


def build_bulk_chart_data(df: pd.DataFrame) -> dict:
    """Charts built from real input rows — no synthetic data needed."""
    def hist_bins(series, n=30):
        counts, edges = np.histogram(series.dropna(), bins=n)
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
        return [round(float(x), 2) for x in centers], [int(c) for c in counts]

    cat_rgba = {"Good": "rgba(240,192,64,", "Medium": "rgba(0,212,255,", "Bad": "rgba(231,76,60,"}
    scatter_pts = {}
    for cat, rgba in cat_rgba.items():
        sub = df[df["Category"] == cat]
        if len(sub):
            scatter_pts[cat] = {
                "x": [round(float(v), 2) for v in sub["SQN_Score"]],
                "y": [round(float(v), 2) for v in sub["Net_Profit"]],
            }

    cat_counts = df["Category"].value_counts().reindex(
        ["Good", "Medium", "Bad"], fill_value=0
    )
    return {
        "histograms": {
            "net_profit": dict(zip(["x", "y"], hist_bins(df["Net_Profit"]))),
            "drawdown":   dict(zip(["x", "y"], hist_bins(df["Drawdown"]))),
            "win_rate":   dict(zip(["x", "y"], hist_bins(df["Win_Rate"]))),
            "sqn_score":  dict(zip(["x", "y"], hist_bins(df["SQN_Score"]))),
        },
        "scatter": scatter_pts,
        "donut": {
            "labels": ["Good", "Medium", "Bad"],
            "values": [int(cat_counts["Good"]),
                       int(cat_counts["Medium"]),
                       int(cat_counts["Bad"])],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-STRATEGY ROBUSTNESS SCORING  (Monte Carlo based)
# ═══════════════════════════════════════════════════════════════════════════════

def _piecewise_score(value: float, points: list[tuple[float, float]]) -> float:
    """Linear interpolation helper for 0–100 component scores."""
    value = float(value or 0)
    points = sorted(points, key=lambda x: x[0])
    if value <= points[0][0]:
        return float(points[0][1])
    if value >= points[-1][0]:
        return float(points[-1][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / max(x1 - x0, 1e-9)
            return float(y0 + t * (y1 - y0))
    return 0.0


def _grade_from_score(score: float) -> str:
    return "A" if score >= 80 else \
           "B" if score >= 65 else \
           "C" if score >= 50 else \
           "D" if score >= 35 else "F"


def _recommendation_from_score(score: float) -> str:
    if score >= 80:
        return "READY FOR DEMO TESTING"
    if score >= 65:
        return "PROMISING — NEEDS MORE VALIDATION"
    if score >= 50:
        return "OPTIMIZE IN SQX BEFORE DEMO"
    return "NOT READY"


def compute_robustness(df_synth: pd.DataFrame, curves: dict) -> dict:
    """
    Institutional-style robustness score for a single uploaded strategy.

    Weights:
      - SQN Quality:                 30%
      - Return / Drawdown Quality:   25%
      - Drawdown Control:            15%
      - Profitability Consistency:   15%
      - Monte Carlo Survival:        15%

    Weak SQN is now a hard drag on the final score.
    """
    avg_sqn       = float(df_synth["SQN_Score"].mean())
    median_ret_dd = float(df_synth["Ret_DD_Ratio"].median())
    median_profit = float(df_synth["Net_Profit"].median())
    median_dd     = float(df_synth["Drawdown"].median())
    positive_pct  = float((df_synth["Net_Profit"] > 0).mean() * 100)
    negative_pct  = 100.0 - positive_pct
    good_pct      = float((df_synth["Category"] == "Good").mean() * 100)

    final_vals = np.array([float(c[-1]) for c in curves.values()], dtype=float)
    survival_pct = float((final_vals > 0).mean() * 100)
    p05_final    = float(np.percentile(final_vals, 5))

    sqn_quality = _piecewise_score(avg_sqn, [
        (0.00, 0), (1.00, 5), (1.50, 25), (2.00, 55),
        (2.50, 78), (3.00, 92), (3.50, 100),
    ])

    ret_dd_quality = _piecewise_score(median_ret_dd, [
        (0.00, 0), (1.00, 15), (2.00, 40), (3.00, 62),
        (5.00, 85), (8.00, 100),
    ])

    if median_profit <= 0:
        dd_to_profit = 999.0
        drawdown_control = 0.0
    else:
        dd_to_profit = median_dd / max(abs(median_profit), 1e-9)
        if dd_to_profit <= 0.20:
            drawdown_control = 95.0
        elif dd_to_profit <= 0.35:
            drawdown_control = 82.0
        elif dd_to_profit <= 0.50:
            drawdown_control = 65.0
        elif dd_to_profit <= 0.75:
            drawdown_control = 45.0
        elif dd_to_profit <= 1.00:
            drawdown_control = 25.0
        elif dd_to_profit <= 1.50:
            drawdown_control = 10.0
        else:
            drawdown_control = 0.0

    profitability_consistency = _piecewise_score(positive_pct, [
        (0, 0), (35, 15), (50, 35), (65, 58), (80, 78),
        (90, 92), (97, 100),
    ])

    monte_carlo_survival = _piecewise_score(survival_pct, [
        (0, 0), (50, 30), (65, 55), (80, 75), (90, 90), (97, 100),
    ])
    if p05_final < 0:
        monte_carlo_survival *= 0.75
    if negative_pct > 35:
        monte_carlo_survival *= 0.55
        profitability_consistency *= 0.65

    sqn_cap = 100.0
    if avg_sqn < 1.50:
        sqn_cap = 58.0
    elif avg_sqn < 2.00:
        sqn_cap = 72.0

    components = {
        "sqn_quality": round(sqn_quality, 1),
        "ret_dd_quality": round(ret_dd_quality, 1),
        "drawdown_control": round(drawdown_control, 1),
        "profitability_consistency": round(profitability_consistency, 1),
        "monte_carlo_survival": round(monte_carlo_survival, 1),

        "profit_stability": round(profitability_consistency, 1),
        "risk_efficiency": round(ret_dd_quality, 1),
        "category_health": round(good_pct, 1),
        "curve_consistency": round(monte_carlo_survival, 1),
    }

    composite = (
        0.30 * sqn_quality +
        0.25 * ret_dd_quality +
        0.15 * drawdown_control +
        0.15 * profitability_consistency +
        0.15 * monte_carlo_survival
    )
    composite = min(composite, sqn_cap)
    composite = round(float(np.clip(composite, 0, 100)), 1)
    grade = _grade_from_score(composite)
    recommendation = _recommendation_from_score(composite)

    reasons = []
    if avg_sqn < 1.5:
        reasons.append("SQN is below 1.5, so statistical quality is weak.")
    elif avg_sqn < 2.0:
        reasons.append("SQN is between 1.5 and 2.0, so the strategy needs caution.")
    elif avg_sqn >= 2.5:
        reasons.append("SQN is strong across the synthetic tests.")

    if median_ret_dd < 2:
        reasons.append("Return/DD is below 2, which is weak for deployment.")
    elif median_ret_dd >= 5:
        reasons.append("Return/DD is strong, showing good reward versus drawdown.")

    if negative_pct > 35:
        reasons.append("More than 35% of Monte Carlo paths finished negative.")
    elif survival_pct >= 90:
        reasons.append("Most Monte Carlo paths survived with positive ending equity.")

    if dd_to_profit > 0.75:
        reasons.append("Median drawdown is high compared with median profit.")

    if not reasons:
        reasons.append("The strategy shows balanced robustness, but still needs demo validation before live use.")

    return {
        "composite": composite,
        "grade": grade,
        "recommendation": recommendation,
        "explanation": " ".join(reasons),
        "components": components,
        "stats": {
            "avg_sqn":       round(avg_sqn, 2),
            "positive_pct":  round(positive_pct, 1),
            "negative_pct":  round(negative_pct, 1),
            "median_ret_dd": round(median_ret_dd, 2),
            "median_drawdown": round(median_dd, 2),
            "dd_to_profit":  round(dd_to_profit, 3) if dd_to_profit < 998 else None,
            "survival_pct":  round(survival_pct, 1),
            "p05_final":     round(p05_final, 2),
            "good_pct":      round(good_pct, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SINGLE-STRATEGY CHART DATA BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _rnd(arr):
    return [round(float(x), 2) for x in arr]


def build_chart_data(df_synth: pd.DataFrame, curves: dict, df_input: pd.DataFrame) -> dict:
    curve_len = max(len(v) for v in curves.values())
    labels    = list(range(curve_len))

    cat_rgba = {
        "Good":   "rgba(240,192,64,",
        "Medium": "rgba(0,212,255,",
        "Bad":    "rgba(231,76,60,",
    }

    # Equity fan: up to 80 curves per category + median line per category
    equity_datasets = []
    for cat, rgba in cat_rgba.items():
        ids = df_synth[df_synth["Category"] == cat]["Strategy_ID"].tolist()
        for sid in ids[:80]:
            c = curves[sid]
            # Pad shorter curves to full length
            if len(c) < curve_len:
                c = np.concatenate([c, np.full(curve_len - len(c), c[-1])])
            equity_datasets.append({
                "data": _rnd(c),
                "borderColor": rgba + "0.18)",
                "borderWidth": 1,
                "pointRadius": 0,
                "tension": 0.3,
                "fill": False,
                "category": cat,
            })
    for cat, rgba in cat_rgba.items():
        ids = df_synth[df_synth["Category"] == cat]["Strategy_ID"].tolist()
        if not ids:
            continue
        mat = []
        for sid in ids:
            c = curves[sid]
            if len(c) < curve_len:
                c = np.concatenate([c, np.full(curve_len - len(c), c[-1])])
            mat.append(c)
        median_curve = np.median(np.array(mat), axis=0)
        equity_datasets.append({
            "label": f"{cat} Median",
            "data": _rnd(median_curve),
            "borderColor": rgba + "1)",
            "borderWidth": 2.5,
            "pointRadius": 0,
            "tension": 0.3,
            "fill": False,
        })

    def hist_bins(series, n=30):
        counts, edges = np.histogram(series.dropna(), bins=n)
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
        return _rnd(centers), [int(c) for c in counts]

    scatter_pts = {
        cat: {
            "x": _rnd(df_synth[df_synth["Category"] == cat]["SQN_Score"]),
            "y": _rnd(df_synth[df_synth["Category"] == cat]["Net_Profit"]),
        }
        for cat in ["Good", "Medium", "Bad"]
    }

    cat_counts = df_synth["Category"].value_counts().reindex(
        ["Good", "Medium", "Bad"], fill_value=0
    )

    comparison = {}
    for col in NUMERIC_COLS:
        inp_vals = df_input[col].dropna().tolist() if col in df_input.columns else []
        syn_vals = df_synth[col].tolist()
        comparison[col] = {
            "input_mean": round(float(np.mean(inp_vals)), 2) if inp_vals else None,
            "synth_mean": round(float(np.mean(syn_vals)), 2),
            "input_std":  round(float(np.std(inp_vals)),  2) if inp_vals else None,
            "synth_std":  round(float(np.std(syn_vals)),  2),
        }

    return {
        "equity": {"labels": labels, "datasets": equity_datasets},
        "histograms": {
            "net_profit": dict(zip(["x","y"], hist_bins(df_synth["Net_Profit"]))),
            "drawdown":   dict(zip(["x","y"], hist_bins(df_synth["Drawdown"]))),
            "win_rate":   dict(zip(["x","y"], hist_bins(df_synth["Win_Rate"]))),
            "sqn_score":  dict(zip(["x","y"], hist_bins(df_synth["SQN_Score"]))),
        },
        "scatter": scatter_pts,
        "donut": {
            "labels": ["Good", "Medium", "Bad"],
            "values": [int(cat_counts["Good"]),
                       int(cat_counts["Medium"]),
                       int(cat_counts["Bad"])],
        },
        "comparison": comparison,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


def _make_bulk_response(metric_rows: list[dict], filename: str, fmt: str, trade_map: dict[str, np.ndarray] | None = None) -> dict:
    """Build the bulk JSON payload from a list of per-strategy metric dicts."""
    df_input  = pd.DataFrame(metric_rows)
    df_norm   = normalize_summary(df_input)
    bulk_strategies, bulk_summary = build_bulk_results(df_norm, trade_map=trade_map)
    score  = build_bulk_score(df_norm, bulk_strategies)
    charts = build_bulk_chart_data(df_norm)
    return {
        "ok":         True,
        "mode":       "bulk",
        "input_mode": fmt,
        "bulk":       {"strategies": bulk_strategies, "summary": bulk_summary},
        "score":      score,
        "charts":     charts,
        "meta": {
            "filename":   filename,
            "input_rows": len(df_norm),
            "synth_rows": 0,
            "n_trades":   0,
            "format":     fmt,
        },
    }


@app.route("/api/analyse", methods=["POST"])
def analyse_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    files = request.files.getlist("file")
    files = [f for f in files if f.filename.lower().endswith(".csv")]
    if not files:
        return jsonify({"error": "Only CSV files are accepted"}), 400

    # ── MULTI-FILE PATH: each file is one strategy trade log ──────────────────
    if len(files) > 1:
        metric_rows = []
        trade_map = {}
        for f in files:
            try:
                raw = f.read()
                fmt, df, _ = detect_and_parse(raw)
                if fmt == "sqx_trade_log":
                    pnl, _, _ = parse_sqx_trade_log(df)
                    if len(pnl) >= 10:
                        name = Path(f.filename).stem
                        metric_rows.append(trades_to_metrics(pnl, name))
                        trade_map[name] = pnl
                elif fmt in ("summary", "statement"):
                    pass  # skip non-trade-log files in a multi-upload
            except Exception:
                pass

        if len(metric_rows) < 2:
            return jsonify({"error": "Could not extract at least 2 valid strategy trade logs."}), 400

        return jsonify(_make_bulk_response(
            metric_rows,
            filename=f"{len(metric_rows)} strategies",
            fmt="multi_file",
            trade_map=trade_map,
        ))

    # ── SINGLE FILE PATH ──────────────────────────────────────────────────────
    f   = files[0]
    raw = f.read()
    text = raw.decode("utf-8", errors="replace")

    # ── Pre-check: concatenated multi-strategy export (repeated headers) ──────
    # Do this BEFORE detect_and_parse so that a file whose first line is a
    # strategy name (not the CSV header) is handled correctly.
    try:
        segments = split_concatenated_sqx(text)
        if len(segments) >= 2:
            metric_rows = []
            trade_map = {}
            # Determine separator from the actual CSV header
            header_line = segments[0][1].split("\n")[0] if segments else ""
            sep = ";" if header_line.count(";") > header_line.count(",") else ","
            for strat_name, seg_text in segments:
                try:
                    seg_df = pd.read_csv(
                        io.StringIO(seg_text), sep=sep, quotechar='"',
                        on_bad_lines="skip", dtype=str,
                    )
                    seg_df = _clean_df(seg_df)
                    pnl, _, _ = parse_sqx_trade_log(seg_df)
                    if len(pnl) >= 10:
                        metric_rows.append(trades_to_metrics(pnl, strat_name))
                        trade_map[strat_name] = pnl
                except Exception:
                    pass
            if len(metric_rows) >= 2:
                return jsonify(_make_bulk_response(
                    metric_rows, filename=f.filename, fmt="sqx_multi", trade_map=trade_map
                ))
    except Exception:
        pass

    try:
        fmt, df, info = detect_and_parse(raw)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400

    # ── format-specific pipeline ───────────────────────────────────────────────
    try:
        input_mode = fmt

        if fmt == "sqx_trade_log":
            # ── Check for Comment-column strategy grouping ────────────────────
            comment_groups = group_by_comment(df)
            if comment_groups:
                metric_rows = [
                    trades_to_metrics(pnl, name)
                    for name, pnl in comment_groups.items()
                ]
                if len(metric_rows) >= 2:
                    return jsonify(_make_bulk_response(
                        metric_rows, filename=f.filename, fmt="sqx_comment", trade_map=comment_groups
                    ))

            # ── Single strategy ───────────────────────────────────────────────
            trade_pnl, balance, sample_type = parse_sqx_trade_log(df)
            if len(trade_pnl) < 10:
                return jsonify({"error": "Too few trades in file (minimum 10)."}), 400

            input_metrics = trades_to_metrics(trade_pnl, "REAL")
            df_input      = pd.DataFrame([input_metrics])
            df_synth, curves = bootstrap_single_strategy(trade_pnl)

            # Include IST/OOS split info if available
            oos_info = None
            if sample_type is not None:
                st_norm = sample_type.str.upper()
                ist_mask = st_norm == "IST"
                oos_mask = ~ist_mask
                if oos_mask.any() and ist_mask.any():
                    ist_metrics = trades_to_metrics(trade_pnl[ist_mask.to_numpy()], "IST")
                    oos_metrics = trades_to_metrics(trade_pnl[oos_mask.to_numpy()], "OOS")
                    oos_info = {"ist": ist_metrics, "oos": oos_metrics}

        elif fmt == "statement":
            strategy_map = parse_statement(df)

            if len(strategy_map) >= 2:
                metric_rows = [trades_to_metrics(pnl, name)
                               for name, pnl in strategy_map.items()]
                return jsonify(_make_bulk_response(
                    metric_rows, filename=f.filename, fmt="statement", trade_map=strategy_map
                ))

            # Single strategy → bootstrap
            name, trade_pnl = max(strategy_map.items(), key=lambda kv: len(kv[1]))
            input_metrics = trades_to_metrics(trade_pnl, name)
            df_input      = pd.DataFrame([input_metrics])
            df_synth, curves = bootstrap_single_strategy(trade_pnl)
            oos_info = None

        elif fmt == "summary":
            df_norm = normalize_summary(df)
            missing = [c for c in NUMERIC_COLS + ["Category"]
                       if c not in df_norm.columns]
            if missing:
                return jsonify({
                    "error": f"Could not map these required columns: {missing}. "
                             f"Found columns: {list(df.columns)}"
                }), 400
            if len(df_norm) < 1:
                return jsonify({"error": "No valid strategy rows found."}), 400

            # ── BULK MODE: 2+ strategy rows ────────────────────────────────────
            if len(df_norm) >= 2:
                return jsonify(_make_bulk_response(
                    df_norm.to_dict("records"), filename=f.filename, fmt="summary"
                ))

            # ── single-row summary: parametric simulation ──────────────────────
            df_input = df_norm
            cat_props    = df_norm["Category"].value_counts(normalize=True).to_dict()
            stats_by_cat = {}
            for cat, grp in df_norm.groupby("Category"):
                sub = grp[NUMERIC_COLS]
                stats_by_cat[cat] = {
                    "mean": sub.mean(), "std": sub.std().fillna(sub.mean()*0.10),
                    "min":  sub.min(),  "max": sub.max(),
                }
            df_synth = generate_params_from_summary(stats_by_cat, cat_props)
            curves   = build_curves_from_summary(df_synth)
            oos_info = None

        else:
            return jsonify({
                "error": "Unrecognised file format.",
                "hint":  "Accepted formats: (1) SQX trade log — semicolon-separated with "
                         "Profit/Loss column; (2) multiple SQX trade logs in one file (header "
                         "repeats between strategies); (3) live statement with Profit + Action "
                         "columns; (4) summary table with Net_Profit/SQN_Score/Category columns. "
                         "You can also select/drop multiple CSV files at once.",
                "found_columns": info.get("columns", []),
            }), 400

        # ── shared scoring & charting ──────────────────────────────────────────
        score  = compute_robustness(df_synth, curves)
        charts = build_chart_data(df_synth, curves, df_input)

        response = {
            "ok":          True,
            "mode":        "single",
            "input_mode":  input_mode,
            "score":       score,
            "charts":      charts,
            "meta": {
                "filename":     f.filename,
                "input_rows":   len(df_input),
                "synth_rows":   len(df_synth),
                "n_trades":     int(max(len(v) - 1 for v in curves.values())),
                "format":       fmt,
            },
        }
        if fmt == "sqx_trade_log" and oos_info:
            response["oos_comparison"] = oos_info

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)

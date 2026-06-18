"""
Real-data Stage-1 builder for the NOTES §3.2 factor-recovery experiment
(credential-free preliminary; swaps to WRDS/CRSP later without touching the model).

Mapping of §3.2's  R_i = U_i T F_i  onto real data:
  - U_i  : per-period firm descriptors -> OpenAssetPricing firm characteristics
           (valuations / growth / risk attributes, exactly §3.2's examples),
           taken as-of each period's start (lagged, no look-ahead).
  - R    : returns of the same d firms -> daily Yahoo-Finance total returns.
  - period i : a calendar year. Characteristics are ~annual, so one U_i per year;
           the ~252 daily return vectors in that year are the observations that
           share that loading (the F-draws under a fixed U_i).
  - T, F : latent in real data (no ground truth) -> not emitted. Validation moves
           to covariance/subspace recovery + factor comparison, not loading_recovery.

Balanced panel: §3.2's U buffer is (P, d, m) -- the SAME d firms in every period --
so we keep only firms with continuous coverage on both sides over the window.

Emits the same .npz contract as generate_factor_data.py (minus T/F/beta):
  R (N,d) f32, period_idx (N,) i64, U/Q/V (P,d,m)/(P,d,m)/(P,m,m) f64,
  plus tickers, permnos, years, char_names, d/m/k/periods.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch

import config.config as config

warnings.filterwarnings("ignore")

CACHE = "data/cache"
DEFAULT_CHARS = ["BM", "Mom12m", "GP", "AssetGrowth", "OperProf"]


# ----------------------------- data acquisition ----------------------------- #

def load_oap_chars(permnos, chars):
    """Long (permno, yyyymm, char) OAP characteristics for the given permnos.

    Reads the cached parquet files written by the download step; falls back to
    downloading via the openassetpricing package if a char is not cached."""
    frames = []
    need = []
    for c in chars:
        p_single = f"{CACHE}/oap_{c}.parquet"
        p_multi = f"{CACHE}/oap_chars.parquet"
        if os.path.exists(p_single):
            df = pd.read_parquet(p_single)
            frames.append(df[df.permno.isin(permnos)][["permno", "yyyymm", c]])
        elif os.path.exists(p_multi) and c in pd.read_parquet(p_multi, columns=None).columns:
            df = pd.read_parquet(p_multi)
            frames.append(df[df.permno.isin(permnos)][["permno", "yyyymm", c]])
        else:
            need.append(c)
    if need:
        import openassetpricing as oap
        df = oap.OpenAP().dl_signal("pandas", need)
        df = df[df.permno.isin(permnos)]
        for c in need:
            frames.append(df[["permno", "yyyymm", c]])
    # outer-merge all chars on (permno, yyyymm)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["permno", "yyyymm"], how="outer")
    return out.sort_values(["permno", "yyyymm"]).reset_index(drop=True)


def load_yahoo_returns(tickers, start, end):
    """Daily simple total returns, (dates x tickers), cached to parquet."""
    key = f"{CACHE}/yahoo_{start}_{end}_{len(tickers)}.parquet"
    if os.path.exists(key):
        px = pd.read_parquet(key)
    else:
        import yfinance as yf
        raw = yf.download(list(tickers), start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        px = raw["Close"].copy()
        os.makedirs(CACHE, exist_ok=True)
        px.to_parquet(key)
    px = px.sort_index()
    rets = px.pct_change().iloc[1:]
    return rets


# ------------------------------ panel assembly ------------------------------ #

def build_panel(crosswalk_csv, chars, start_year, end_year, k):
    xw = pd.read_csv(crosswalk_csv)
    tickers = xw.ticker.tolist()
    permnos = xw.permno.tolist()
    permno_of = dict(zip(xw.ticker, xw.permno))

    years = list(range(start_year, end_year + 1))
    P = len(years)

    # --- characteristics: as-of Dec(y-1), forward-filled, per firm --- #
    co = load_oap_chars(permnos, chars)                      # long
    # wide per (permno, yyyymm); reindex monthly, forward-fill within firm
    co = co.set_index(["permno", "yyyymm"]).sort_index()
    # U_year[permno] = char vector using latest value at or before Dec(year-1)
    asof = {}           # (permno) -> DataFrame indexed by year with m chars
    for permno in permnos:
        if permno not in co.index.get_level_values(0):
            continue
        sub = co.loc[permno].copy()                          # index yyyymm
        sub = sub.sort_index().ffill()                       # carry last known value
        rows = {}
        for y in years:
            cutoff = (y - 1) * 100 + 12                       # Dec of prior year
            avail = sub[sub.index <= cutoff]
            if len(avail) and avail.iloc[-1][chars].notna().all():
                rows[y] = avail.iloc[-1][chars].values.astype(float)
        asof[permno] = rows

    # --- returns: daily, per ticker --- #
    rets = load_yahoo_returns(tickers, f"{start_year}-01-01", f"{end_year}-12-31")

    # --- choose the balanced universe: firms with chars for ALL years and
    #     full daily-return coverage every year --- #
    keep = []
    for t in tickers:
        pn = permno_of[t]
        has_chars = pn in asof and all(y in asof[pn] for y in years)
        has_rets = t in rets.columns and (
            rets[t].notna().groupby(rets.index.year).sum().reindex(years).fillna(0) > 100
        ).all()
        if has_chars and has_rets:
            keep.append(t)
        else:
            why = []
            if not has_chars:
                miss = [] if pn not in asof else [y for y in years if y not in asof[pn]]
                why.append(f"chars missing yrs={miss[:4]}{'...' if len(miss) > 4 else ''}")
            if not has_rets:
                why.append("sparse/again returns")
            print(f"  drop {t:<6} ({'; '.join(why)})")

    d = len(keep)
    m = len(chars)
    assert k <= m <= d, f"need k<=m<=d, got k={k}, m={m}, d={d} (universe too small)"
    print(f"\nBalanced universe: d={d} firms, m={m} chars, k={k}, P={P} years "
          f"({start_year}-{end_year})")
    print("  firms:", " ".join(keep))

    keep_pn = [permno_of[t] for t in keep]

    # --- U: (P, d, m), cross-sectionally z-scored per period (BARRA-style) --- #
    U = np.zeros((P, d, m))
    for pi, y in enumerate(years):
        M = np.vstack([asof[pn][y] for pn in keep_pn])       # (d, m) raw chars
        mu, sd = M.mean(0, keepdims=True), M.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        U[pi] = (M - mu) / sd                                # standardized descriptors

    # --- R: stack daily return vectors, label by period (year) --- #
    R_rows, idx_rows = [], []
    for pi, y in enumerate(years):
        ry = rets.loc[rets.index.year == y, keep].dropna(how="any")
        R_rows.append(ry.values)
        idx_rows.append(np.full(len(ry), pi, dtype=np.int64))
    R = np.vstack(R_rows)
    period_idx = np.concatenate(idx_rows)

    # global scalar so mean per-asset variance ~ 1 (no per-asset standardizing,
    # which would distort U_i; a single scale folds harmlessly into F's scale)
    scale = np.sqrt((R.var(0)).mean())
    R = R / scale

    return dict(R=R, period_idx=period_idx, U=U, tickers=keep, permnos=keep_pn,
                years=years, chars=chars, d=d, m=m, k=k, P=P, ret_scale=scale)


# ------------------------------ QR + save ----------------------------------- #

def finalize(panel, seed):
    U = torch.from_numpy(panel["U"]).double()
    Q, V = torch.linalg.qr(U, mode="reduced")
    panel["Q"], panel["V"] = Q.numpy(), V.numpy()

    # rank check (thin QR needs rank-m U_i)
    ranks = torch.linalg.matrix_rank(U).tolist()
    print(f"\n  rank(U_i) per period: {ranks}  (need all = m = {panel['m']})")
    if not all(r == panel["m"] for r in ranks):
        print("  *** WARNING: some U_i rank-deficient; descriptors collinear ***")

    var = panel["R"].var(0)
    print(f"  per-asset return variance: mean={var.mean():.3f} "
          f"min={var.min():.3f} max={var.max():.3f} (target ~1)")
    print(f"  N={len(panel['R'])} daily obs, ret_scale={panel['ret_scale']:.5f}")
    panel["seed"] = seed
    return panel


def save(panel, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        R=panel["R"].astype(np.float32),
        period_idx=panel["period_idx"].astype(np.int64),
        U=panel["U"], Q=panel["Q"], V=panel["V"],
        d=panel["d"], m=panel["m"], k=panel["k"], periods=panel["P"],
        obs_per_period=len(panel["R"]) // panel["P"],
        seed=panel["seed"],
        tickers=np.array(panel["tickers"]), permnos=np.array(panel["permnos"]),
        years=np.array(panel["years"]), char_names=np.array(panel["chars"]),
        ret_scale=panel["ret_scale"],
    )
    print(f"\nSaved real-data dataset to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build real-data §3.2 panel (Yahoo returns + OAP chars)")
    ap.add_argument("--crosswalk", default="data/ticker_permno_crosswalk.csv")
    ap.add_argument("--chars", nargs="+", default=DEFAULT_CHARS)
    ap.add_argument("--start_year", type=int, default=2005)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--k", type=int, default=3, help="number of latent factors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="empirical_analysis_data/real_factor_period_data.npz")
    args = ap.parse_args()

    config.set_seed(args.seed)
    panel = build_panel(args.crosswalk, args.chars, args.start_year, args.end_year, args.k)
    panel = finalize(panel, args.seed)
    save(panel, args.out)

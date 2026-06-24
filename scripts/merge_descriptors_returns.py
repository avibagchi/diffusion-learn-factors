#!/usr/bin/env python3
"""
Merge Compustat daily price rows (returns.csv) with quarterly descriptor rows
(descriptors.csv).

Daily simple returns are computed from adjusted prices:
    adj_t = prccd * trfd / ajexdi
    ret_t = adj_t / adj_{t-1} - 1

Descriptors are attached with a backward as-of join on (GVKEY, datadate), so each
return day uses the most recent quarterly fundamentals available on or before
that date (no look-ahead).

Outputs:
  - Long merged table (CSV or Parquet): one row per firm-day with return + descriptors
  - Optional §3.2 panel .npz for train_factor.py (--to_npz)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Columns used only for linking / labels — not treated as model descriptors.
METADATA_COLS = {
    "GVKEY", "LINKPRIM", "LIID", "LINKTYPE", "LPERMNO", "LPERMCO",
    "LINKDT", "LINKENDDT", "iid", "datadate", "tic", "conm", "cusip",
    "fyearq", "fqtr", "fyr", "indfmt", "consol", "popsrc", "datafmt",
    "datacqtr", "datafqtr", "finalq", "ogmq", "rp", "scfq", "srcq",
    "staltq", "updq", "apdedateq", "fdateq", "pdateq", "rdq",
    "curcdq", "curncdq", "currtrq", "curuscnq", "acctchgq", "acctstdq",
    "adrrq", "ajexq", "ajpq", "bsprq", "compstq", "exchg", "cik",
    "costat", "fic", "fyrc", "gind", "gsector", "gsubind", "naics",
    "priusa", "sic", "ipodate",
    # Price fields from the returns file (return is computed separately).
    "ajexdi", "cshoc", "cshtrd", "eps", "prccd", "trfd",
    # Derived in this script.
    "adj_price", "return",
}


def _normalize_gvkey(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lstrip("0").replace("", np.nan).astype("Int64")


def infer_descriptor_columns(desc: pd.DataFrame) -> list[str]:
    """Numeric Compustat columns that are not linkage/metadata."""
    cols = []
    for c in desc.columns:
        if c in METADATA_COLS:
            continue
        if pd.api.types.is_numeric_dtype(desc[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric descriptor columns found in descriptors file")
    return cols


def load_returns(path: Path) -> pd.DataFrame:
    usecols = ["GVKEY", "LPERMNO", "datadate", "tic", "prccd", "trfd", "ajexdi"]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    df["GVKEY"] = _normalize_gvkey(df["GVKEY"])
    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.dropna(subset=["GVKEY", "datadate", "prccd", "trfd", "ajexdi"])
    df = df.sort_values(["GVKEY", "datadate"])
    df["adj_price"] = df["prccd"] * df["trfd"] / df["ajexdi"]
    df["return"] = df.groupby("GVKEY", sort=False)["adj_price"].pct_change()
    df = df.dropna(subset=["return"])
    df = df.drop_duplicates(subset=["GVKEY", "datadate"], keep="last")
    return df


def load_descriptors(path: Path, descriptor_cols: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    df["GVKEY"] = _normalize_gvkey(df["GVKEY"])
    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.dropna(subset=["GVKEY", "datadate"])

    if descriptor_cols is None:
        descriptor_cols = infer_descriptor_columns(df)
    else:
        missing = set(descriptor_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Descriptor columns not found: {sorted(missing)}")

    keep = ["GVKEY", "LPERMNO", "datadate", "tic"] + descriptor_cols
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["GVKEY", "datadate"])
    return df, descriptor_cols


def merge_asof(returns: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join: latest quarterly descriptors on or before each return day."""
    ret_cols = ["GVKEY", "LPERMNO", "datadate", "tic", "return"]
    ret_cols = [c for c in ret_cols if c in returns.columns]

    merged_parts = []
    for gvkey, ret_g in returns.groupby("GVKEY", sort=False):
        desc_g = descriptors[descriptors["GVKEY"] == gvkey]
        if desc_g.empty:
            continue
        part = pd.merge_asof(
            ret_g[ret_cols].sort_values("datadate"),
            desc_g.sort_values("datadate"),
            on="datadate",
            by="GVKEY",
            direction="backward",
            suffixes=("", "_desc"),
        )
        merged_parts.append(part)

    if not merged_parts:
        raise ValueError("No overlapping firms between returns and descriptors")

    merged = pd.concat(merged_parts, ignore_index=True)
    return merged.sort_values(["GVKEY", "datadate"]).reset_index(drop=True)


def drop_rows_missing_descriptors(merged: pd.DataFrame, descriptor_cols: list[str]) -> pd.DataFrame:
    mask = merged[descriptor_cols].notna().any(axis=1)
    dropped = int((~mask).sum())
    if dropped:
        print(f"Dropping {dropped:,} rows with no descriptors available yet")
    return merged.loc[mask].reset_index(drop=True)


def save_long_table(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    print(f"Saved merged long table ({len(df):,} rows) to {path}")


def _firm_column(df: pd.DataFrame) -> str:
    """Prefer GVKEY for firm identity; fall back to tic."""
    if "GVKEY" in df.columns and df["GVKEY"].notna().any():
        return "GVKEY"
    if "tic" in df.columns and df["tic"].notna().any():
        return "tic"
    raise ValueError("Merged data must contain GVKEY or tic to identify firms")


def _effective_min_obs(df: pd.DataFrame, years: list[int], min_obs_per_year: int, firm_col: str) -> int:
    """Cap min_obs by the shortest year's typical trading days (median firm obs)."""
    counts = df.groupby([firm_col, "year"]).size()
    year_medians = []
    for y in years:
        if y not in counts.index.get_level_values("year"):
            year_medians.append(0)
            continue
        year_medians.append(float(counts.xs(y, level="year").median()))
    cap = int(min(year_medians)) if year_medians else 0
    effective = min(min_obs_per_year, cap) if cap > 0 else min_obs_per_year
    if effective < min_obs_per_year:
        print(
            f"Lowering min_obs_per_year {min_obs_per_year} -> {effective} "
            f"(median firm obs by year: {dict(zip(years, [int(m) for m in year_medians]))})"
        )
    return max(1, effective)


def _select_balanced_firms(
    df: pd.DataFrame,
    years: list[int],
    descriptor_cols: list[str],
    min_obs: int,
    firm_col: str,
    max_firms: int | None,
) -> list:
    """Firms with enough daily obs and descriptors in every year."""
    obs = df.groupby([firm_col, "year"]).size().unstack(fill_value=0).reindex(columns=years, fill_value=0)
    desc_ok = (
        df.groupby([firm_col, "year"])[descriptor_cols]
        .apply(lambda g: g.notna().any().any())
        .unstack(fill_value=False)
        .reindex(columns=years, fill_value=False)
    )
    eligible = (obs >= min_obs) & desc_ok
    firms = obs.index[eligible.all(axis=1)].tolist()

    if max_firms is not None and len(firms) > max_firms:
        total_obs = obs.loc[firms].sum(axis=1).sort_values(ascending=False)
        firms = total_obs.head(max_firms).index.tolist()
        print(f"Capped balanced panel to top {max_firms} firms by total obs")

    return firms


def _year_return_matrix(
    df: pd.DataFrame,
    year: int,
    firms: list,
    firm_col: str,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """(T, d) daily returns on dates where every firm has an observation."""
    sub = df[(df["year"] == year) & (df[firm_col].isin(firms))]
    wide = sub.pivot_table(index="datadate", columns=firm_col, values="return", aggfunc="first")
    wide = wide.reindex(columns=firms)
    complete = wide.dropna(how="any")
    if complete.empty:
        raise ValueError(
            f"No common trading dates for all {len(firms)} firms in {year}. "
            "Try --max_firms with a smaller value or a narrower year window."
        )
    return complete.to_numpy(dtype=np.float32), complete.index


def build_panel_npz(
    merged: pd.DataFrame,
    descriptor_cols: list[str],
    out_path: Path,
    *,
    start_year: int | None,
    end_year: int | None,
    k: int,
    min_obs_per_year: int,
    max_firms: int | None = None,
) -> None:
    """
    Build the §3.2 panel used by train_factor.py:
      R (N,d), period_idx (N,), U/Q/V (P,d,m), ...
    One calendar year = one period; U_i is cross-sectionally z-scored descriptors
    as-of the first descriptor observation in that year per firm.
    """
    import torch

    df = merged.copy()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["year"] = df["datadate"].dt.year
    firm_col = _firm_column(df)
    df = df.drop_duplicates(subset=[firm_col, "datadate"], keep="last")
    if start_year is not None:
        df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]
    if df.empty:
        raise ValueError("No rows left after year filtering")

    firm_col = _firm_column(df)
    years = sorted(int(y) for y in df["year"].unique())
    min_obs = _effective_min_obs(df, years, min_obs_per_year, firm_col)
    keep = _select_balanced_firms(df, years, descriptor_cols, min_obs, firm_col, max_firms)

    if len(keep) < 2:
        obs = df.groupby([firm_col, "year"]).size().unstack(fill_value=0)
        print("Per-year obs summary (max / median firms meeting threshold):")
        for y in years:
            col = obs[y] if y in obs.columns else pd.Series(dtype=int)
            print(
                f"  {y}: max={int(col.max()) if len(col) else 0}, "
                f"firms>={min_obs}={int((col >= min_obs).sum()) if len(col) else 0}"
            )
        raise ValueError(
            f"Balanced panel too small (d={len(keep)}). "
            "Try lowering --min_obs_per_year, narrowing --start_year/--end_year, "
            "or set --max_firms."
        )

    d = len(keep)
    m = len(descriptor_cols)
    P = len(years)
    if k > m:
        raise ValueError(f"Need k <= m <= d, got k={k}, m={m}, d={d}")

    print(f"Balanced panel: d={d} firms, m={m} descriptors, P={P} years, k={k} "
          f"(firm id: {firm_col}, min_obs={min_obs})")

    U = np.zeros((P, d, m), dtype=np.float64)
    R_rows, idx_rows = [], []

    for pi, y in enumerate(years):
        chars = np.zeros((d, m), dtype=np.float64)
        for di, firm in enumerate(keep):
            sub = df[(df[firm_col] == firm) & (df["year"] == y)].sort_values("datadate")
            vals = sub.iloc[0][descriptor_cols].to_numpy(dtype=np.float64)
            chars[di] = np.nan_to_num(vals, nan=0.0)
        mu, sd = chars.mean(0, keepdims=True), chars.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        U[pi] = (chars - mu) / sd

        R_y, _ = _year_return_matrix(df, y, keep, firm_col)
        R_rows.append(R_y)
        idx_rows.append(np.full(len(R_y), pi, dtype=np.int64))

    R = np.vstack(R_rows)
    period_idx = np.concatenate(idx_rows)
    scale = float(np.sqrt(R.var(0).mean()))
    if scale > 0:
        R = R / scale

    U_t = torch.from_numpy(U).double()
    Q, V = torch.linalg.qr(U_t, mode="reduced")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        R=R.astype(np.float32),
        period_idx=period_idx.astype(np.int64),
        U=U,
        Q=Q.numpy(),
        V=V.numpy(),
        d=d,
        m=m,
        k=k,
        periods=P,
        obs_per_period=len(R) // P,
        tickers=np.array([str(x) for x in keep]),
        years=np.array(years),
        char_names=np.array(descriptor_cols),
        ret_scale=scale,
        source="merge_descriptors_returns.py",
    )
    meta = {
        "num_samples": int(len(R)),
        "num_assets": d,
        "num_descriptors": m,
        "num_factors": k,
        "num_periods": P,
        "years": years,
        "firm_id": firm_col,
        "tickers": [str(x) for x in keep],
        "descriptor_cols": descriptor_cols,
        "min_obs_per_year": min_obs,
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved panel .npz to {out_path}  (N={len(R)}, d={d}, m={m}, P={P})")


def main():
    parser = argparse.ArgumentParser(description="Merge returns.csv with descriptors.csv")
    parser.add_argument("--returns", type=Path, default=Path("data/returns.csv"))
    parser.add_argument("--descriptors", type=Path, default=Path("data/descriptors.csv"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/merged_returns_descriptors.parquet"),
        help="Long merged output path (.csv or .parquet)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default=None,
        help="Output format (default: inferred from --out suffix)",
    )
    parser.add_argument(
        "--descriptor_cols",
        nargs="+",
        default=None,
        help="Descriptor columns to keep (default: all numeric non-metadata cols)",
    )
    parser.add_argument(
        "--keep_missing_descriptors",
        action="store_true",
        help="Keep return rows even if no descriptor is available yet",
    )
    parser.add_argument("--to_npz", type=Path, default=None, help="Also write §3.2 panel .npz here")
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)
    parser.add_argument("--k", type=int, default=3, help="Latent factors (stored in .npz only)")
    parser.add_argument(
        "--min_obs_per_year",
        type=int,
        default=100,
        help="Min daily obs per firm-year (auto-lowered if a year has fewer trading days)",
    )
    parser.add_argument(
        "--max_firms",
        type=int,
        default=100,
        help="Max firms in balanced panel (top by total obs; 0 = no cap)",
    )
    args = parser.parse_args()

    fmt = args.format or args.out.suffix.lstrip(".")
    if fmt not in {"csv", "parquet"}:
        fmt = "parquet"

    print(f"Loading returns from {args.returns} ...")
    returns = load_returns(args.returns)
    print(f"  {len(returns):,} daily return rows, {returns['GVKEY'].nunique():,} firms")

    print(f"Loading descriptors from {args.descriptors} ...")
    descriptors, descriptor_cols = load_descriptors(args.descriptors, args.descriptor_cols)
    print(f"  {len(descriptors):,} descriptor rows, {len(descriptor_cols)} descriptor columns")

    print("Merging (backward as-of on GVKEY + datadate) ...")
    merged = merge_asof(returns, descriptors)
    if not args.keep_missing_descriptors:
        merged = drop_rows_missing_descriptors(merged, descriptor_cols)

    save_long_table(merged, args.out, fmt)

    if args.to_npz is not None:
        max_firms = None if args.max_firms == 0 else args.max_firms
        build_panel_npz(
            merged,
            descriptor_cols,
            args.to_npz,
            start_year=args.start_year,
            end_year=args.end_year,
            k=args.k,
            min_obs_per_year=args.min_obs_per_year,
            max_firms=max_firms,
        )


if __name__ == "__main__":
    main()

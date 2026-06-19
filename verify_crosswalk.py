"""
Identity-verified ticker->permno crosswalk builder (credential-free, pre-WRDS).

OAP ships no ticker, so a hand-built crosswalk is needed -- and memorized CRSP
permnos are ~50% wrong. We verify each candidate (ticker, permno) two ways
against data we already have, so a wrong permno is dropped instead of silently
polluting the balanced panel:

  1. IDENTITY:  OAP's Mom12m (12-month momentum) is derived from the firm's own
                returns. corr( OAP.Mom12m[permno] , Yahoo trailing-12m[ticker] )
                is ~0.95 for the right permno and far lower for a different firm.
  2. COVERAGE:  all m descriptors present, ~continuously, every year in-window.

Survivors are written to a verified crosswalk CSV consumed by build_real_data.py.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CACHE = "data/cache"
CHARS = ["BM", "Mom12m", "GP", "AssetGrowth", "OperProf"]

# (ticker, best-guess CRSP permno, name) -- large, pre-2005, non-financial.
# Wrong permnos are expected; the Mom12m check filters them out.
CANDIDATES = [
    # the 9 already verified (sanity re-check)
    ("AAPL", 14593, "Apple"), ("MSFT", 10107, "Microsoft"), ("WMT", 55976, "Walmart"),
    ("INTC", 59328, "Intel"), ("CVX", 14541, "Chevron"), ("CAT", 18542, "Caterpillar"),
    ("CSCO", 76076, "Cisco"), ("TXN", 15579, "Texas Instruments"), ("COST", 87055, "Costco"),
    # new large-cap candidates
    ("KO", 11308, "Coca-Cola"), ("PEP", 13856, "PepsiCo"), ("PG", 18163, "Procter & Gamble"),
    ("JNJ", 22111, "Johnson & Johnson"), ("PFE", 21936, "Pfizer"), ("MRK", 22752, "Merck"),
    ("XOM", 11850, "Exxon Mobil"), ("MCD", 43449, "McDonald's"), ("HD", 66181, "Home Depot"),
    ("IBM", 12490, "IBM"), ("ORCL", 10104, "Oracle"), ("QCOM", 77178, "Qualcomm"),
    ("AMGN", 57665, "Amgen"), ("MMM", 22592, "3M"), ("HON", 10145, "Honeywell"),
    ("BA", 19561, "Boeing"), ("GE", 12060, "General Electric"), ("T", 10401, "AT&T"),
    ("ABT", 20482, "Abbott"), ("MDT", 69796, "Medtronic"), ("LLY", 50876, "Eli Lilly"),
    ("NKE", 57033, "Nike"), ("SBUX", 75510, "Starbucks"), ("UNP", 32707, "Union Pacific"),
    ("DE", 22517, "Deere"), ("EMR", 22779, "Emerson"), ("CL", 20587, "Colgate"),
    ("KMB", 21573, "Kimberly-Clark"), ("ADP", 16548, "ADP"), ("AMAT", 30681, "Applied Materials"),
    ("ADBE", 75241, "Adobe"), ("MU", 33724, "Micron"), ("GILD", 75909, "Gilead"),
    ("BMY", 19393, "Bristol-Myers"), ("AMD", 61241, "AMD"), ("TGT", 76281, "Target"),
    ("ADI", 60397, "Analog Devices"), ("LRCX", 75154, "Lam Research"), ("LOW", 21178, "Lowe's"),
    ("UPS", 89093, "UPS"), ("SLB", 13928, "Schlumberger"), ("DHR", 50778, "Danaher"),
    ("NSC", 21020, "Norfolk Southern"), ("ITW", 22760, "Illinois Tool Works"),
    ("ETN", 22871, "Eaton"), ("APD", 21809, "Air Products"), ("SYK", 78758, "Stryker"),
    ("PH", 21207, "Parker Hannifin"), ("ROK", 80415, "Rockwell"), ("PCAR", 47626, "Paccar"),
]


def load_oap(permnos):
    bm = pd.read_parquet(f"{CACHE}/oap_BM.parquet")
    ch = pd.read_parquet(f"{CACHE}/oap_chars.parquet")
    df = bm.merge(ch, on=["permno", "yyyymm"], how="outer")
    return df[df.permno.isin(permnos)]


def yahoo_monthly(tickers, start, end):
    key = f"{CACHE}/yahoo_verify_{start}_{end}_{len(tickers)}.parquet"
    if os.path.exists(key):
        px = pd.read_parquet(key)
    else:
        import yfinance as yf
        raw = yf.download(list(tickers), start=start, end=end, interval="1mo",
                          auto_adjust=True, progress=False, threads=False)
        px = raw["Close"].copy()
        os.makedirs(CACHE, exist_ok=True)
        px.to_parquet(key)
    return px.sort_index()


def main():
    ap = argparse.ArgumentParser(description="Build identity-verified crosswalk")
    ap.add_argument("--start_year", type=int, default=2005)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--corr_thresh", type=float, default=0.75)
    ap.add_argument("--min_year_cov", type=float, default=0.9,
                    help="fraction of in-window years that must have all chars")
    ap.add_argument("--out", default="data/ticker_permno_crosswalk_verified.csv")
    args = ap.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    permnos = [p for _, p, _ in CANDIDATES]
    tickers = [t for t, _, _ in CANDIDATES]

    oap = load_oap(permnos).sort_values(["permno", "yyyymm"])
    px = yahoo_monthly(tickers, f"{args.start_year - 1}-01-01", f"{args.end_year}-12-31")
    tr12_all = px.pct_change(12)
    tr12_all.index = tr12_all.index.year * 100 + tr12_all.index.month

    print(f"{'ticker':<6}{'permno':>8}{'corr':>8}{'char_cov':>10}  verdict")
    print("-" * 50)
    kept = []
    for t, pn, name in CANDIDATES:
        sub = oap[oap.permno == pn]
        # identity: Mom12m vs Yahoo trailing-12m
        mom = sub[["yyyymm", "Mom12m"]].dropna().set_index("yyyymm")["Mom12m"]
        corr = np.nan
        if t in tr12_all.columns and len(mom):
            j = pd.concat([mom, tr12_all[t].rename("yh")], axis=1, join="inner").dropna()
            if len(j) > 24:
                corr = j["Mom12m"].corr(j["yh"])
        # coverage: fraction of in-window years with all chars present, using the
        # SAME as-of-Dec(y-1) + forward-fill logic as build_real_data.py, so a KEEP
        # here is guaranteed to survive the build's balanced-panel filter.
        cov = 0.0
        if len(sub):
            s = sub.set_index("yyyymm").sort_index()[CHARS].ffill()
            have = 0
            for y in years:
                avail = s[s.index <= (y - 1) * 100 + 12]
                if len(avail) and avail.iloc[-1].notna().all():
                    have += 1
            cov = have / len(years)
        ok = (corr >= args.corr_thresh) and (cov >= args.min_year_cov)
        verdict = "KEEP" if ok else ("drop:id" if not (corr >= args.corr_thresh) else "drop:cov")
        print(f"{t:<6}{pn:>8}{corr:>8.3f}{cov:>10.2f}  {verdict}")
        if ok:
            kept.append((t, pn, name))

    out = pd.DataFrame(kept, columns=["ticker", "permno", "name"])
    out.to_csv(args.out, index=False)
    print(f"\nVerified d={len(kept)} firms -> {args.out}")
    print(" ".join(t for t, _, _ in kept))


if __name__ == "__main__":
    main()

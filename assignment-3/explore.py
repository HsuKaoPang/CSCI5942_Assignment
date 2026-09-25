#!/usr/bin/env python3
"""Try a mix filter against a sample of the corpus, before spending a
catalog+train cycle on it.

    python explore.py                                  # overview only
    python explore.py "stop_hits >= 3 AND ends_sentence"
    python explore.py "upper_ratio < 0.3" --shards 8

The filter string takes the same SQL-ish form as catalog.py's MIXES, so
whatever reads well here can be pasted straight into that dict.
"""

import argparse
import glob
import re

import pandas as pd
import pyarrow.parquet as pq

BUDGET = 50_000_000
ALL_SHARDS = 64


def load(corpus: str, n_shards: int):
    paths = sorted(glob.glob(f"{corpus}/*.parquet"))[:n_shards]
    if not paths:
        raise SystemExit(f"no parquet shards in {corpus}")
    return pq.read_table(paths).to_pandas(), len(paths)


def to_pandas_query(where: str) -> str:
    """SQL-ish MIXES syntax -> pandas .query() syntax."""
    out = re.sub(r"\bAND\b", "and", where)
    out = re.sub(r"\bOR\b", "or", out)
    out = re.sub(r"\bNOT\b", "not", out)
    return re.sub(r"(?<![<>!=])=(?!=)", "==", out)


def overview(df: pd.DataFrame) -> None:
    """The feature table the report quotes. Run with --shards 64 for the
    numbers over the whole corpus rather than a sample."""
    print(f"{len(df):,} lines, {df['n_chars'].sum() / 1e6:.1f}M chars\n")
    numeric = df.select_dtypes("number").drop(columns=["n_tokens"],
                                              errors="ignore")
    stats = pd.DataFrame({
        "min": numeric.min(),
        "median": numeric.median(),
        "mean": numeric.mean(),
        "max": numeric.max(),
    })
    print(stats.round(2).to_string())
    if "ends_sentence" in df:
        print(f"\nends_sentence: True for "
              f"{100 * df['ends_sentence'].mean():.1f}% of lines")
    print("\nthe maxima are where the junk hides -- chase anything extreme")


def report(df: pd.DataFrame, where: str, scale: float) -> None:
    kept = df.query(to_pandas_query(where))
    dropped = df.drop(kept.index)

    kept_chars = kept["n_chars"].sum()
    est = kept_chars * scale
    verdict = "OK" if est >= BUDGET else f"SHORT of the {BUDGET / 1e6:.0f}M budget"

    print(f"filter: {where}\n")
    print(f"  kept    {len(kept):>9,} lines ({100 * len(kept) / len(df):5.1f}%)"
          f"  mean {kept['n_chars'].mean():6.0f} chars")
    print(f"  dropped {len(dropped):>9,} lines ({100 * len(dropped) / len(df):5.1f}%)"
          f"  mean {dropped['n_chars'].mean():6.0f} chars")
    # The shards are not interchangeable -- the first few hold more
    # English than average -- so a small sample overestimates. Against
    # catalog.py's exact 259.2M for "doc_stopwords >= 5", 2 shards said
    # 323M, 8 said 277M, 16 said 267M. Treat anything within ~20% of the
    # budget as undecided and re-run with more shards.
    print(f"\n  {100 * kept_chars / df['n_chars'].sum():.1f}% of characters"
          f"  ->  ~{est / 1e6:.1f}M tokens extrapolated from"
          f" {int(ALL_SHARDS / scale)}/{ALL_SHARDS} shards  [{verdict}]")

    # Sample rather than .head(): the shards are not in a meaningful
    # order, so the first rows of a page-level filter's output are
    # whatever happened to be at the top of the file, which reads as far
    # worse than the filter actually is. Weight by n_chars too -- the
    # model spends its budget per character, not per line, so a
    # line-weighted sample over-represents the short lines.
    for label, frame in (("KEPT", kept), ("DROPPED", dropped)):
        print(f"\n  --- {label}: random lines ---")
        for text in frame["text"].sample(min(4, len(frame)), random_state=0):
            print(f"    {text[:88]!r}")
        print(f"  --- {label}: random, weighted by characters ---")
        weighted = frame.sample(min(4, len(frame)), weights=frame["n_chars"],
                                random_state=0)
        for text in weighted["text"]:
            print(f"    {text[:88]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("where", nargs="?", help="filter, e.g. 'stop_hits >= 3 AND ends_sentence'")
    ap.add_argument("--corpus", default="out/raw/corpus")
    ap.add_argument("--shards", type=int, default=8,
                    help=f"how many of the {ALL_SHARDS} shards to sample "
                         "(default 8; fewer is faster but overestimates)")
    args = ap.parse_args()

    df, n = load(args.corpus, args.shards)
    if args.where:
        report(df, args.where, ALL_SHARDS / n)
    else:
        overview(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

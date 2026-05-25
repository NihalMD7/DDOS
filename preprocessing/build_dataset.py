"""Materialize the cleaned, label-collapsed dataset from raw CICDDoS2019 shards.

Reads every `*.parquet` under `data/`, drops inf/NaN rows, collapses the
`DrDoS_*` label variants into a single class per attack family, and writes the
combined frame to `artifacts/cicddos2019_clean.parquet` so the training notebook
and scripts can skip the full reload.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
ART_DIR = PROJECT / "artifacts"


def build(min_class_count: int = 20) -> pd.DataFrame:
    ART_DIR.mkdir(exist_ok=True)
    paths = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no parquet shards under {DATA_DIR}")

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df["Label"] = df["Label"].str.replace("^DrDoS_", "", regex=True)
    df["Label"] = df["Label"].str.replace("UDP-lag", "UDPLag", regex=False)

    counts = df["Label"].value_counts()
    keep = counts[counts >= min_class_count].index
    df = df[df["Label"].isin(keep)].reset_index(drop=True)

    out = ART_DIR / "cicddos2019_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows, {df['Label'].nunique()} classes -> {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-class-count", type=int, default=20)
    args = p.parse_args()
    build(args.min_class_count)

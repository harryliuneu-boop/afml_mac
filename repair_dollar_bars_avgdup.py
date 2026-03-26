"""
把 dollar bars 里重复的 timestamp 行合并为一行，并对数值列取平均。

输入：parquet（timestamp 可能在索引或列里）
输出：带后缀的 repaired parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PARQUET = Path("outputs/bars_from_daily_trades_/dollar_10000000.parquet")
OUTPUT_PARQUET = Path("outputs/bars_from_daily_trades_/dollar_10000000_repaired_avgdup.parquet")


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"找不到文件: {INPUT_PARQUET}")

    df = pd.read_parquet(INPUT_PARQUET)
    if df.empty:
        raise RuntimeError("输入 parquet 为空。")

    # 兼容 timestamp 在索引/列两种情况
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        df2 = df.copy()
        df2["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
        df2 = df2.dropna(subset=["timestamp"])
        group_key = "timestamp"
    else:
        if df.index.name != "timestamp":
            raise KeyError("无法定位 timestamp：既不在列里，也不是索引名。")
        df2 = df.reset_index()
        df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True, errors="coerce")
        df2 = df2.dropna(subset=["timestamp"])
        group_key = "timestamp"

    # 只对数值列取平均（timestamp 不参与）
    numeric_cols = [c for c in df2.columns if c != group_key and pd.api.types.is_numeric_dtype(df2[c])]
    if not numeric_cols:
        raise RuntimeError("没有可用于平均的数值列。")

    repaired = (
        df2.groupby(group_key, as_index=False)[numeric_cols]
        .mean()
    )

    repaired = repaired.sort_values(group_key).reset_index(drop=True)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    print(f"saved: {OUTPUT_PARQUET} (rows={len(repaired)})")


if __name__ == "__main__":
    main()


"""
分析并绘制 dollar bars：
1) 横轴时间，纵轴 dollar（每根 bar 的美元成交额；通常接近阈值）
2) 横轴时间，纵轴 volume（每根 bar 的成交量，更有分析意义）
3) 统计每周产生多少根 bar（bars/week）

输出：
- outputs/plots/dollar_bars/dollar_over_time.png
- outputs/plots/dollar_bars/volume_over_time.png
- outputs/plots/dollar_bars/bars_per_week.png
- outputs/plots/dollar_bars/bars_per_week.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 你当前生成的 dollar bars 文件（可按需修改）
DOLLAR_BARS_PATH = Path("/Users/liuhaoran/Documents/program/afml/outputs/bars_from_daily_trades_test/dollar_dynamic_k288.parquet")

# 输出目录
OUT_DIR = Path("outputs/plots/dollar_bars")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not DOLLAR_BARS_PATH.exists():
        raise FileNotFoundError(f"找不到文件: {DOLLAR_BARS_PATH}")

    ensure_dir(OUT_DIR)

    df = pd.read_parquet(DOLLAR_BARS_PATH)
    if df.empty:
        raise RuntimeError("dollar bars 文件为空。")

    if "timestamp" not in df.columns:
        raise KeyError("缺少 timestamp 列。")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 1) dollar 随时间
    if "dollar" in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df["timestamp"], df["dollar"].astype(float), linewidth=0.8)
        plt.title("Dollar bars: dollar per bar over time")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Dollar per bar")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = OUT_DIR / "dollar_over_time.png"
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"saved: {out}")
    else:
        print("skip: no 'dollar' column")

    # 2) volume 随时间
    if "volume" in df.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df["timestamp"], df["volume"].astype(float), linewidth=0.8)
        plt.title("Dollar bars: volume per bar over time")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Volume per bar")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = OUT_DIR / "volume_over_time.png"
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"saved: {out}")
    else:
        print("skip: no 'volume' column")

    # 3) 每周 bar 数量
    week_start = df["timestamp"].dt.to_period("W").dt.start_time.dt.tz_localize("UTC")
    bars_per_week = df.groupby(week_start).size().rename("bars").to_frame()
    bars_csv = OUT_DIR / "bars_per_week.csv"
    bars_per_week.to_csv(bars_csv)
    print(f"saved: {bars_csv}")

    plt.figure(figsize=(12, 4))
    plt.plot(bars_per_week.index, bars_per_week["bars"], marker="o", linewidth=1.0, markersize=3)
    plt.title("Dollar bars: bars per week")
    plt.xlabel("Week start (UTC)")
    plt.ylabel("Bars / week")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "bars_per_week.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()


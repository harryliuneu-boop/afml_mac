"""
根据 time bars 的“bar 数量”推导 volume/dollar bars 的阈值。

目标：
- 读取最近 1 年（默认到昨天为止，不含今天）逐日 trades zip
- 对给定时间频率（5m, 15m, 30m, 1h, 2h, 4h, 1d, 7d）计算 time bars 的总根数
- 同时统计期间总 dollar 与总 qty
- 输出：
  - dollar_per_bar = total_dollar / time_bar_count(freq)
  - vol_per_bar     = total_qty    / time_bar_count(freq)

这相当于让 volume/dollar bars 在“平均”上产生与 time bars 相近数量的分桶尺度。
注意：不同类型 bars 的分割规则不同，因此每一天不一定严格相等，但平均阈值是对齐粒度的实用方法。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from crypto_bar_analysis import build_time_bars


# ============================================================
# 固定配置区（你可按需修改）
# ============================================================
INPUT_DIR = Path("/Users/liuhaoran/Documents/program/afml/download/spot/daily/trades/BTCUSDT/")
OUTPUT_CSV = Path("/Users/liuhaoran/Documents/program/afml/outputs/bar_thresholds_from_timebar_counts.csv")

LOOKBACK_YEARS = 1
END_DATE: date | None = None  # None => UTC 昨天

TIME_FREQS = ["5m", "15m", "30m", "1h", "2h", "4h", "1d", "7d"]

THRESHOLD_STAT_NOTE = "time_bar_count_based"

DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})")


# ============================================================
# 工具函数
# ============================================================
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def normalize_time_freq(freq: str) -> str:
    """
    兼容 pandas 新版本频率别名（避免 FutureWarning）。
    - 5m/15m/30m -> 5min/15min/30min
    """
    f = freq.strip().lower()
    m = re.fullmatch(r"(\d+)\s*m", f)
    if m:
        return f"{m.group(1)}min"
    return f


def extract_day_from_name(path: Path) -> date | None:
    m = DATE_PAT.search(path.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1), tz="UTC").date()


def iter_zip_files(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*.zip"):
        if str(p).endswith(".zip.tmp"):
            continue
        yield p


def read_single_zip_trades(path: Path) -> pd.DataFrame:
    """
    返回列：timestamp, price, qty, dollar
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 7:
        raise ValueError(f"不支持的列数: {path} -> {df.shape[1]} 列")
    df.columns = [
        "trade_id",
        "price",
        "qty",
        "quoteQty",
        "time",
        "isBuyerMaker",
        "isBestMatch",
    ]
    df = df[["trade_id", "price", "qty", "quoteQty", "time"]].copy()
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["dollar"] = pd.to_numeric(df["quoteQty"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price", "qty", "dollar"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["trade_id"], keep="last")
    return df[["timestamp", "price", "qty", "dollar"]]


def get_date_range(lookback_years: int, end_date: date | None) -> tuple[date, date]:
    if end_date is None:
        end_date = pd.Timestamp.now(tz="UTC").date() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(lookback_years * 365.25))
    return start_date, end_date


@dataclass
class Row:
    freq: str
    time_bar_count: int
    total_dollar: float
    total_qty: float
    dollar_per_bar: float
    vol_per_bar: float
    bars_per_day_avg: float


def main() -> None:
    start_d, end_d = get_date_range(LOOKBACK_YEARS, END_DATE)
    print(f"=== range === start_d={start_d} end_d={end_d}")

    # 准备 zip 列表（按文件名日期筛）
    zip_files: list[Path] = []
    for p in iter_zip_files(INPUT_DIR):
        d = extract_day_from_name(p)
        if d is None:
            continue
        if start_d <= d <= end_d:
            zip_files.append(p)
    zip_files = sorted(zip_files)

    if not zip_files:
        raise RuntimeError(f"没有找到指定区间内的 zip：{INPUT_DIR}")
    print(f"zip_files={len(zip_files)}")

    time_bar_count: dict[str, int] = {f: 0 for f in TIME_FREQS}

    total_dollar = 0.0
    total_qty = 0.0

    # 将 freq 提前 normalize，避免重复转换
    freq_norm_map = {f: normalize_time_freq(f) for f in TIME_FREQS}

    for i, z in enumerate(zip_files, start=1):
        day_df = read_single_zip_trades(z)
        if day_df.empty:
            continue

        # totals
        total_dollar += float(day_df["dollar"].sum())
        total_qty += float(day_df["qty"].sum())

        # time bars count (逐日 resample，再 count)
        for freq_raw in TIME_FREQS:
            freq_norm = freq_norm_map[freq_raw]
            bars = build_time_bars(day_df, freq=freq_norm)
            # dropna close 后才计入
            time_bar_count[freq_raw] += len(bars)

        if i % 10 == 0 or i == len(zip_files):
            print(f"[progress] {i}/{len(zip_files)} zip | totals: dollar={total_dollar:.3f} qty={total_qty:.3f}")

        del day_df

    # 生成输出
    n_days = (end_d - start_d).days + 1
    rows: list[Row] = []
    for f in TIME_FREQS:
        cnt = time_bar_count[f]
        if cnt <= 0:
            dollar_per_bar = float("nan")
            vol_per_bar = float("nan")
            bars_per_day_avg = 0.0
        else:
            dollar_per_bar = total_dollar / cnt
            vol_per_bar = total_qty / cnt
            bars_per_day_avg = cnt / n_days
        rows.append(
            Row(
                freq=f,
                time_bar_count=int(cnt),
                total_dollar=float(total_dollar),
                total_qty=float(total_qty),
                dollar_per_bar=float(dollar_per_bar),
                vol_per_bar=float(vol_per_bar),
                bars_per_day_avg=float(bars_per_day_avg),
            )
        )

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    out_df["method_note"] = THRESHOLD_STAT_NOTE
    ensure_parent(OUTPUT_CSV)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(out_df)


if __name__ == "__main__":
    main()


"""
根据时间频率（如 1h、15m、1d）反推 volume/dollar bars 的每根阈值。

思想：
1) 用最近 N 年的数据统计每一天的总成交美元/成交量（sum quoteQty / sum qty）。
2) 选一个你希望 volume/dollar bars 近似产生的“bars/day”，用时间频率的 bars/day 作为目标：
   - 1d: 1 根/天
   - 1h: 24 根/天
   - 15m: 96 根/天
   - 4h: 6 根/天
3) 阈值取：threshold = daily_stat_total / bars_per_day

注意：
- 这是“让阈值在长期平均上产生接近目标 bars/day”，并不保证每一天都严格等于目标根数。
- 建议用 median 而不是 mean，抗极端日更稳健。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


INPUT_DIR = Path("/Users/liuhaoran/Documents/program/afml/download/spot/daily/trades/BTCUSDT/")
OUTPUT_CSV = Path("/Users/liuhaoran/Documents/program/afml/outputs/bar_thresholds_from_timefreq.csv")

LOOKBACK_YEARS = 1
END_DATE = None  # None => 用昨天(UTC date)

FREQS = ["1d", "1h", "15m"]  # 支持形式：Nd/Nh/Nm

THRESHOLD_STAT = "median"  # "median" 或 "mean"

PARQUET_COMPRESSION = "snappy"  # 这里不直接用 parquet，只保留作为风格一致

DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})")


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def extract_day_from_name(path: Path) -> date | None:
    m = DATE_PAT.search(path.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1), tz="UTC").date()


def iter_zip_files(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*.zip"):
        # 避免读 tmp
        if str(p).endswith(".zip.tmp"):
            continue
        yield p


def read_day_totals_from_zip(zip_path: Path) -> tuple[float, float]:
    """
    返回： (daily_total_dollar, daily_total_qty)
    """
    df = pd.read_csv(zip_path, header=None)
    if df.shape[1] != 7:
        raise ValueError(f"不支持的列数: {zip_path} -> {df.shape[1]} 列")
    df.columns = ["trade_id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"]
    # 只需要 sum，不必做去重/排序（减少内存）
    df_qty = pd.to_numeric(df["qty"], errors="coerce")
    df_dollar = pd.to_numeric(df["quoteQty"], errors="coerce")
    df_qty = df_qty.dropna()
    df_dollar = df_dollar.dropna()
    return float(df_dollar.sum()), float(df_qty.sum())


def get_date_range(lookback_years: int, end_date: date | None) -> tuple[date, date]:
    if end_date is None:
        end_date = (pd.Timestamp.now(tz="UTC").date() - timedelta(days=1))
    start_date = end_date - timedelta(days=int(lookback_years * 365.25))
    return start_date, end_date


def parse_freq_to_bars_per_day(freq: str) -> float:
    """
    解析 Nd/Nh/Nm 为 bars/day。
    例如：
    - 1d => 1
    - 1h => 24
    - 15m => 96
    """
    s = freq.strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([dhm])", s)
    if not m:
        raise ValueError(f"不支持的 freq 格式: {freq}，仅支持 Nd/Nh/Nm，例如 1d/4h/15m")
    n = float(m.group(1))
    unit = m.group(2)
    if n <= 0:
        raise ValueError(f"freq 数值必须为正：{freq}")

    if unit == "d":
        # 例如 7d: 约 1/7 根/天（一般你不用于 volume/dollar 的目标，但这里支持）
        return 1.0 / n
    if unit == "h":
        return 24.0 / n
    if unit == "m":
        return 1440.0 / n
    raise ValueError(f"未知单位: {unit}")


@dataclass(frozen=True)
class ThresholdRow:
    freq: str
    bars_per_day: float
    daily_dollar_stat: float
    daily_qty_stat: float
    dollar_per_bar: float
    vol_per_bar: float


def stat_func(values: pd.Series, mode: str) -> float:
    mode = mode.lower().strip()
    if mode == "median":
        return float(values.median())
    if mode == "mean":
        return float(values.mean())
    raise ValueError("THRESHOLD_STAT 只能是 median 或 mean")


def main() -> None:
    start_d, end_d = get_date_range(LOOKBACK_YEARS, END_DATE)
    print(f"=== 读取范围 ===")
    print(f"start_d={start_d} end_d={end_d} (lookback_years={LOOKBACK_YEARS})")

    zip_files = []
    for p in iter_zip_files(INPUT_DIR):
        d = extract_day_from_name(p)
        if d is None:
            continue
        if start_d <= d <= end_d:
            zip_files.append(p)
    zip_files = sorted(zip_files)
    print(f"zip_files={len(zip_files)}")

    daily_dollar = {}
    daily_qty = {}
    for i, z in enumerate(zip_files, start=1):
        d = extract_day_from_name(z)
        if d is None:
            continue
        try:
            tot_dollar, tot_qty = read_day_totals_from_zip(z)
        except Exception as e:
            print(f"[warn] 读取失败: {z} -> {e}")
            continue
        daily_dollar[pd.Timestamp(d, tz="UTC")] = tot_dollar
        daily_qty[pd.Timestamp(d, tz="UTC")] = tot_qty
        if i % 10 == 0 or i == len(zip_files):
            print(f"[progress] {i}/{len(zip_files)} processed")

    if not daily_dollar or not daily_qty:
        raise RuntimeError("没有有效的 daily totals（可能输入目录为空或文件格式不一致）")

    s_dollar = pd.Series(daily_dollar).sort_index()
    s_qty = pd.Series(daily_qty).sort_index()

    dollar_stat = stat_func(s_dollar, THRESHOLD_STAT)
    qty_stat = stat_func(s_qty, THRESHOLD_STAT)
    print(f"\n=== daily totals stat ===")
    print(f"{THRESHOLD_STAT} daily total dollar = {dollar_stat:.3f}")
    print(f"{THRESHOLD_STAT} daily total qty    = {qty_stat:.6f}")

    rows: list[ThresholdRow] = []
    for freq in FREQS:
        bars_per_day = parse_freq_to_bars_per_day(freq)
        dollar_per_bar = dollar_stat / bars_per_day
        vol_per_bar = qty_stat / bars_per_day
        rows.append(
            ThresholdRow(
                freq=freq,
                bars_per_day=bars_per_day,
                daily_dollar_stat=dollar_stat,
                daily_qty_stat=qty_stat,
                dollar_per_bar=dollar_per_bar,
                vol_per_bar=vol_per_bar,
            )
        )

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    ensure_parent(OUTPUT_CSV)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(out_df)


if __name__ == "__main__":
    main()


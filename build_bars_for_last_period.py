"""
构建最近一段时间（如最近 1 个月）的 bars（time/volume/dollar），输出 parquet。

重点解决两个问题：
1) 在 dollar bars / volume bars 中，如何设置阈值（dollar_per_bar / vol_per_bar）
2) 如何读取最近日期范围内的按天 zip 原始 trades 数据

阈值自动建议（默认启用）：
- 用「目标每日日志/成交条数」来反推阈值：
  - dollar_per_bar = median(daily_total_dollar) / target_dollar_bars_per_day
  - vol_per_bar    = median(daily_total_qty)    / target_volume_bars_per_day

这样阈值会随币价变化而自适应，且比手工猜一个固定数更稳健。
"""

from __future__ import annotations

import gc
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from crypto_bar_analysis import build_time_bars


# ============================================================
# 固定参数区（按需修改）
# ============================================================
INPUT_DIR = Path("/Users/liuhaoran/Documents/program/afml/download/spot/daily/trades/BTCUSDT/")
OUTPUT_DIR = Path("/Users/liuhaoran/Documents/program/afml/outputs/bars_last_period/")

# 时间范围：最近 N 个月（用 30*N 天近似）
LOOKBACK_MONTHS = 1

# 避免不完整的今天：默认到“昨天(UTC day)”为止
END_DATE = None  # None 表示自动用 date.today()-1
FILTER_ZIP_BY_DATE = True

# 三类 bars 都做
BAR_TYPES = ["time", "volume", "dollar"]

# time bars 周期（pandas freq）
# 注意：避免 5m 这种别名 FutureWarning，建议使用 5min/15min/30min
TIME_FREQS = ["5min", "15min", "30min", "1h", "4h", "1d", "7d"]

# 供自动推导阈值用：目标“每日日志条数”
TARGET_DOLLAR_BARS_PER_DAY = 80
TARGET_VOLUME_BARS_PER_DAY = 120

# 阈值选择方法
THRESHOLD_STAT = "median"  # median 或 mean

# 下限保护，避免极端日导致阈值太小（可根据需要调大）
MIN_DOLLAR_PER_BAR = 1e5
MIN_VOL_PER_BAR = 1.0

# parquet 压缩
PARQUET_COMPRESSION = "snappy"

DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})")


# ============================================================
# 工具函数
# ============================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def discover_zip_files(input_dir: Path) -> list[Path]:
    files = sorted(p for p in input_dir.rglob("*.zip") if not str(p).endswith(".zip.tmp"))
    return files


def extract_day_from_name(path: Path) -> pd.Timestamp | None:
    m = DATE_PAT.search(path.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1), tz="UTC")


def read_single_zip_trades(path: Path) -> pd.DataFrame:
    """
    读取单日 trades zip（Binance daily trades 常见 7 列）。
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


def get_date_range() -> tuple[date, date]:
    if END_DATE is None:
        end = date.today() - timedelta(days=1)
    else:
        end = END_DATE
    start = end - timedelta(days=int(LOOKBACK_MONTHS * 30))
    return start, end


def filter_zip_files(zip_files: list[Path], start_d: date, end_d: date) -> list[Path]:
    out: list[Path] = []
    for z in zip_files:
        d = extract_day_from_name(z)
        if d is None:
            continue
        dd = d.date()
        if dd < start_d or dd > end_d:
            continue
        out.append(z)
    return out


def summarize_daily_totals(zip_files: list[Path]) -> tuple[pd.Series, pd.Series]:
    """
    只做阈值推导需要的统计：每一天的 total dollar / total qty。
    输出：
    - daily_dollar_total: index=day(UTC), value=当天所有 trade 的 dollar 总和
    - daily_vol_total:    index=day(UTC), value=当天所有 trade 的 qty 总和
    """
    daily_dollar: dict[pd.Timestamp, float] = {}
    daily_qty: dict[pd.Timestamp, float] = {}

    for z in zip_files:
        day = extract_day_from_name(z)
        if day is None:
            continue
        day_key = pd.Timestamp(day.date(), tz="UTC")
        t = read_single_zip_trades(z)
        if t.empty:
            continue
        daily_dollar[day_key] = float(t["dollar"].sum())
        daily_qty[day_key] = float(t["qty"].sum())
        del t
        gc.collect()

    if not daily_dollar:
        raise RuntimeError("没有统计到 daily totals（可能 zip 为空或格式不匹配）")

    s_dollar = pd.Series(daily_dollar).sort_index()
    s_qty = pd.Series(daily_qty).sort_index()
    return s_dollar, s_qty


def stat_func(values: pd.Series, mode: str) -> float:
    mode = mode.lower().strip()
    if mode == "median":
        return float(values.median())
    if mode == "mean":
        return float(values.mean())
    raise ValueError("THRESHOLD_STAT 只能是 median 或 mean")


class ValueBarStreamBuilder:
    """正序流式构建 volume/dollar bars（支持跨天延续未完成 bar）。"""

    def __init__(self, target_value: float, value_col: Literal["qty", "dollar"]) -> None:
        if target_value <= 0:
            raise ValueError("target_value 必须为正")
        self.target_value = float(target_value)
        self.value_col = value_col
        self.rows: list[dict] = []

        self.filled = 0.0
        self.open_ts: pd.Timestamp | None = None
        self.high = 0.0
        self.low = 0.0
        self.open_p = 0.0
        self.close_p = 0.0
        self.vol_sum = 0.0
        self.dollar_sum = 0.0
        self.close_ts: pd.Timestamp | None = None

    def feed_df(self, df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            ts = row.timestamp
            price = float(row.price)
            qty = float(row.qty)
            dollar = float(row.dollar)
            val_total = qty if self.value_col == "qty" else dollar
            if val_total <= 0:
                continue

            remaining = val_total
            while remaining > 1e-12:
                need = self.target_value - self.filled
                take = min(remaining, need)
                frac = take / val_total
                q_take = qty * frac
                d_take = dollar * frac

                if self.filled == 0:
                    self.open_ts = ts
                    self.open_p = self.high = self.low = self.close_p = price
                    self.vol_sum = q_take
                    self.dollar_sum = d_take
                else:
                    self.high = max(self.high, price)
                    self.low = min(self.low, price)
                    self.close_p = price
                    self.vol_sum += q_take
                    self.dollar_sum += d_take

                self.filled += take
                remaining -= take
                self.close_ts = ts

                if self.filled >= self.target_value - 1e-9:
                    self._flush_full()

    def _flush_full(self) -> None:
        assert self.open_ts is not None and self.close_ts is not None
        self.rows.append(
            {
                "timestamp": self.close_ts,
                "open": self.open_p,
                "high": self.high,
                "low": self.low,
                "close": self.close_p,
                "volume": self.vol_sum,
                "dollar": self.dollar_sum,
            }
        )
        self.filled = 0.0
        self.open_ts = None
        self.close_ts = None
        self.vol_sum = 0.0
        self.dollar_sum = 0.0

    def finalize(self) -> pd.DataFrame:
        # 与前面脚本逻辑一致：最后一根未满且不足半阈值则丢弃
        if self.filled > 0:
            if self.filled >= self.target_value * 0.5 and self.open_ts is not None and self.close_ts is not None:
                self._flush_full()
        out = pd.DataFrame(self.rows)
        return out.reset_index(drop=True)


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    df.to_parquet(out_path, index=False, engine="pyarrow", compression=PARQUET_COMPRESSION)


def merge_time_bar_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    if out.empty:
        return out
    out = out.sort_values("timestamp")
    out = out.groupby("timestamp", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        dollar=("dollar", "sum"),
    )
    return out.reset_index(drop=True)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    zip_files = discover_zip_files(INPUT_DIR)
    if not zip_files:
        raise FileNotFoundError(f"输入目录未找到 zip：{INPUT_DIR}")

    start_d, end_d = get_date_range()
    if FILTER_ZIP_BY_DATE:
        zip_use = filter_zip_files(zip_files, start_d=start_d, end_d=end_d)
    else:
        zip_use = zip_files

    zip_use = sorted(zip_use)
    if not zip_use:
        raise RuntimeError("过滤后没有 zip 可处理。请检查 LOOKBACK_MONTHS / END_DATE / 输入路径。")

    print("=== Input ===")
    print(f"start_d={start_d} end_d={end_d} zip_count={len(zip_use)}")

    # 1) 推导阈值
    daily_dollar_total, daily_qty_total = summarize_daily_totals(zip_use)

    median_dollar = stat_func(daily_dollar_total, THRESHOLD_STAT)
    median_qty = stat_func(daily_qty_total, THRESHOLD_STAT)

    dollar_per_bar = max(float(median_dollar) / TARGET_DOLLAR_BARS_PER_DAY, MIN_DOLLAR_PER_BAR)
    vol_per_bar = max(float(median_qty) / TARGET_VOLUME_BARS_PER_DAY, MIN_VOL_PER_BAR)

    print("=== Threshold suggestion ===")
    print(f"{THRESHOLD_STAT} daily dollar total = {median_dollar:.3f}")
    print(f"{THRESHOLD_STAT} daily qty total    = {median_qty:.6f}")
    print(f"dollar_per_bar = {dollar_per_bar:.3f} (target {TARGET_DOLLAR_BARS_PER_DAY} bars/day)")
    print(f"vol_per_bar     = {vol_per_bar:.6f} (target {TARGET_VOLUME_BARS_PER_DAY} bars/day)")

    # 2) 构建 bars（正序）
    time_chunks: dict[str, list[pd.DataFrame]] = {freq: [] for freq in TIME_FREQS}
    vol_builder = ValueBarStreamBuilder(vol_per_bar, "qty") if "volume" in BAR_TYPES else None
    dol_builder = ValueBarStreamBuilder(dollar_per_bar, "dollar") if "dollar" in BAR_TYPES else None

    t0 = pd.Timestamp.now()
    trades_so_far = 0

    for i, z in enumerate(zip_use, start=1):
        day_df = read_single_zip_trades(z)
        if day_df.empty:
            continue
        trades_so_far += len(day_df)

        if "time" in BAR_TYPES:
            for freq in TIME_FREQS:
                day_bars = build_time_bars(day_df, freq=freq)
                if not day_bars.empty:
                    time_chunks[freq].append(day_bars)

        if vol_builder is not None:
            vol_builder.feed_df(day_df)
        if dol_builder is not None:
            dol_builder.feed_df(day_df)

        del day_df
        gc.collect()

        if i % 5 == 0 or i == len(zip_use):
            print(f"[progress] {i}/{len(zip_use)} zip | trades={trades_so_far:,}")

    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    print(f"=== Build done === elapsed={elapsed:.1f}s trades={trades_so_far:,}")

    # 3) 写 parquet
    tag = f"{start_d.isoformat()}_{end_d.isoformat()}"

    if "time" in BAR_TYPES:
        for freq in TIME_FREQS:
            bars = merge_time_bar_chunks(time_chunks[freq])
            out = OUTPUT_DIR / f"time_{freq}_{tag}.parquet"
            write_parquet(bars, out)
            print(f"[time] {freq} rows={len(bars)} -> {out}")

    if "volume" in BAR_TYPES and vol_builder is not None:
        bars = vol_builder.finalize()
        out = OUTPUT_DIR / f"volume_{vol_per_bar:.6f}_{tag}.parquet"
        write_parquet(bars, out)
        print(f"[volume] rows={len(bars)} -> {out}")

    if "dollar" in BAR_TYPES and dol_builder is not None:
        bars = dol_builder.finalize()
        out = OUTPUT_DIR / f"dollar_{dollar_per_bar:.3f}_{tag}.parquet"
        write_parquet(bars, out)
        print(f"[dollar] rows={len(bars)} -> {out}")

    meta = pd.DataFrame(
        [
            {
                "input_dir": str(INPUT_DIR),
                "output_dir": str(OUTPUT_DIR),
                "start_d": str(start_d),
                "end_d": str(end_d),
                "zip_count": len(zip_use),
                "time_freqs": ",".join(TIME_FREQS),
                "target_dollar_bars_per_day": TARGET_DOLLAR_BARS_PER_DAY,
                "target_volume_bars_per_day": TARGET_VOLUME_BARS_PER_DAY,
                "threshold_stat": THRESHOLD_STAT,
                "dollar_per_bar": dollar_per_bar,
                "vol_per_bar": vol_per_bar,
            }
        ]
    )
    write_parquet(meta, OUTPUT_DIR / f"build_meta_{tag}.parquet")
    print(f"meta -> {OUTPUT_DIR / f'build_meta_{tag}.parquet'}")


if __name__ == "__main__":
    main()


"""
从按天分割的 Binance 原始成交 zip 构建 bars（time/volume/dollar），并保存为 parquet。

设计目标：
1) 参数直接写在脚本中（无需命令行）
2) 输入目录读取全部 zip，忽略 .tmp/非 zip
3) 支持配置 bars 列表（time/volume/dollar，默认三种）
4) time bars 支持多周期；volume/dollar 支持多阈值
5) 默认起始时间支持“自动推断”
6) 输出统一保存为 parquet

内存说明（重要）：
- volume / dollar bars 默认用「流式」按 zip 顺序处理，避免一次性 concat 全历史逐笔导致 OOM（zsh: killed）。
- time bars 按「每个 zip 单日」分别 resample 再拼接；跨日对齐与整段 resample 在极少数周期下可能有细微差别，一般可接受。
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd

from crypto_bar_analysis import build_time_bars


# ============================================================
# 固定参数区（按需修改）
# ============================================================
INPUT_DIR = Path("/Users/liuhaoran/Documents/program/afml/download/spot/daily/trades/BTCUSDT/")
OUTPUT_DIR = Path("/Users/liuhaoran/Documents/program/afml/outputs/bars_from_daily_trades_test/")

# 运行模式：
# - "full": 全量重建（按 START_TIME/END_TIME 限制，若都为 None 则处理全部 zip）
# - "incremental": 增量构建（优先用 AUTO_START_MODE 推断起点，只处理新增 zip）
RUN_MODE: Literal["incremental", "full"] = "full"

# 构建哪些 bars：可选 "time", "volume", "dollar"
# BAR_TYPES = ["time", "volume", "dollar"]
BAR_TYPES = ["dollar"]

# time bars 周期（pandas offset：1m, 5m, 15m, 1h, 4h, 1d, 7d 等）
TIME_FREQS = ["5m", "15m", "30m", "1h", "2h", "4h", "1d", "7d"]

# volume / dollar 阈值
VOLUME_THRESHOLDS = [100.0, 500.0]
DOLLAR_THRESHOLDS = [1e7, 5e7]

# volume/dollar 模式：
# - "fixed": 使用固定阈值（VOLUME_THRESHOLDS / DOLLAR_THRESHOLDS）
# - "dynamic": 使用动态阈值（按目标 bars/day + 滚动 + EWMA）
VALUE_BAR_MODE: Literal["fixed", "dynamic"] = "dynamic"

# 动态模式下的目标 bars/day（例如 [288, 96, 48, 24, 12, 6]）
# TARGET_BARS_PER_DAY = [288, 96, 48, 24, 12, 6]
TARGET_BARS_PER_DAY = [288]

# 动态阈值参数
DYNAMIC_ROLLING_DAYS = 30
DYNAMIC_EWMA_ALPHA = 0.2
# 限制阈值的日变化幅度（例如 0.2 表示相对前一日最多 ±20%）
DYNAMIC_MAX_DAILY_CHANGE = 0.2
# 可选阈值上下界（None 表示不限制）
DYNAMIC_MIN_THRESHOLD: float | None = None
DYNAMIC_MAX_THRESHOLD: float | None = None

# 时间过滤（UTC）
START_TIME: str | None = None
END_TIME: str | None = None

# 自动起始时间模式：
# - "last_output_day_end": 读取输出目录里已有 bars parquet 的最大 timestamp，取「次日 00:00:00」作为起点
# - "last_input_day_end": 取输入 zip 文件名中最后一天（不读 parquet）
# - "none": 不自动推断
AUTO_START_MODE = "last_output_day_end"

# 仅使用文件名中的日期筛选 zip（强烈推荐 True，避免读无关历史）
FILTER_ZIP_BY_DATE = True

# 限制最多处理多少个 zip（None=不限制；测试可设 3）
MAX_ZIP_FILES: int | None = 100

# volume/dollar 用流式构建（低内存）；False 则一次性读入全部 zip（易 OOM）
STREAM_VOLUME_DOLLAR = True

# parquet 压缩
PARQUET_COMPRESSION = "snappy"

# 进度输出频率（每处理多少个 zip 输出一次；<=1 表示每个 zip 都输出）
PROGRESS_EVERY_ZIP: int = 1


DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_time_freq(freq: str) -> str:
    """
    兼容 pandas 新版本频率别名：
    - 5m/15m/30m -> 5min/15min/30min
    避免 FutureWarning: 'm' is deprecated...
    """
    f = freq.strip()
    m = re.fullmatch(r"(\d+)\s*m", f, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}min"
    return f


def discover_zip_files(input_dir: Path) -> list[Path]:
    files = sorted(p for p in input_dir.rglob("*.zip") if not str(p).endswith(".zip.tmp"))
    return files


def extract_day_from_name(path: Path) -> pd.Timestamp | None:
    m = DATE_PAT.search(path.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1), tz="UTC")


def read_single_zip_trades(path: Path) -> pd.DataFrame:
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
    # 有少量文件可能出现异常 time 值（超出 datetime64[ns] 范围或单位不匹配）。
    # 使用 errors="coerce" 避免整体任务因单行数据崩溃，把异常时间行置为 NaT 后再 dropna。
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["dollar"] = pd.to_numeric(df["quoteQty"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price", "qty", "dollar"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["trade_id"], keep="last")
    return df[["timestamp", "price", "qty", "dollar"]]


def infer_start_time(output_dir: Path, zip_files: list[Path], mode: str) -> pd.Timestamp | None:
    if mode == "none":
        return None

    if mode == "last_output_day_end":
        parquet_files = sorted(output_dir.rglob("*.parquet"))
        max_ts: pd.Timestamp | None = None
        for p in parquet_files:
            if p.name == "build_meta.parquet":
                continue
            if not (p.name.startswith("time_") or p.name.startswith("volume_") or p.name.startswith("dollar_")):
                continue
            try:
                df = pd.read_parquet(p, columns=["timestamp"])
            except Exception:
                continue
            if df.empty:
                continue
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
            if ts.empty:
                continue
            cur = ts.max()
            if (max_ts is None) or (cur > max_ts):
                max_ts = cur
        if max_ts is None:
            return None
        return max_ts.normalize() + pd.Timedelta(days=1)

    if mode == "last_input_day_end":
        days = [extract_day_from_name(p) for p in zip_files]
        days = [d for d in days if d is not None]
        if not days:
            return None
        last_day = max(days)
        return last_day.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    raise ValueError(f"未知 AUTO_START_MODE: {mode}")


def effective_min_zip_day(
    start_time: str | None,
    auto_start: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """zip 文件名日期的下界（含该日）。"""
    if start_time is not None:
        return pd.Timestamp(start_time, tz="UTC").normalize()
    if auto_start is not None:
        return auto_start.normalize()
    return None


def effective_max_zip_day(end_time: str | None) -> pd.Timestamp | None:
    if end_time is None:
        return None
    return pd.Timestamp(end_time, tz="UTC").normalize()


def filter_zip_files(
    zip_files: list[Path],
    min_day: pd.Timestamp | None,
    max_day: pd.Timestamp | None,
    max_count: int | None,
) -> list[Path]:
    out: list[Path] = []
    for z in zip_files:
        d = extract_day_from_name(z)
        if d is None:
            continue
        if min_day is not None and d.normalize() < min_day.normalize():
            continue
        if max_day is not None and d.normalize() > max_day.normalize():
            continue
        out.append(z)
    if max_count is not None:
        out = out[:max_count]
    return out


def stream_build_value_bars(
    zip_paths: list[Path],
    target_value: float,
    value_col: Literal["qty", "dollar"],
) -> pd.DataFrame:
    """
    按 zip 顺序流式构建 volume/dollar bars，内存近似 O(单日)。
    单笔成交若超过 target，会拆成多根 bar（按比例分摊 qty/dollar，与常见实现一致）。
    """
    if target_value <= 0:
        raise ValueError("target_value 必须为正")

    rows: list[dict] = []
    filled = 0.0
    open_ts: pd.Timestamp | None = None
    high = low = open_p = close_p = 0.0
    vol_sum = 0.0
    dollar_sum = 0.0
    close_ts: pd.Timestamp | None = None

    def flush_partial_if_needed() -> None:
        nonlocal filled, open_ts, high, low, open_p, close_p, vol_sum, dollar_sum, close_ts
        if filled <= 0:
            return
        if filled < target_value * 0.5:
            filled = 0.0
            open_ts = None
            vol_sum = dollar_sum = 0.0
            close_ts = None
            return
        assert close_ts is not None and open_ts is not None
        rows.append(
            {
                "timestamp": close_ts,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close_p,
                "volume": vol_sum,
                "dollar": dollar_sum,
            }
        )
        filled = 0.0
        open_ts = None
        vol_sum = dollar_sum = 0.0
        close_ts = None

    def flush_full_bar() -> None:
        nonlocal filled, open_ts, high, low, open_p, close_p, vol_sum, dollar_sum, close_ts
        assert close_ts is not None and open_ts is not None
        rows.append(
            {
                "timestamp": close_ts,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close_p,
                "volume": vol_sum,
                "dollar": dollar_sum,
            }
        )
        filled = 0.0
        open_ts = None
        vol_sum = dollar_sum = 0.0
        close_ts = None

    for zp in zip_paths:
        df = read_single_zip_trades(zp)
        for row in df.itertuples(index=False):
            ts = row.timestamp
            price = float(row.price)
            qty = float(row.qty)
            dollar = float(row.dollar)
            val_total = qty if value_col == "qty" else dollar
            if val_total <= 0:
                continue
            remaining = val_total
            while remaining > 1e-12:
                need = target_value - filled
                take = min(remaining, need)
                frac = take / val_total
                q_take = qty * frac
                d_take = dollar * frac

                if filled == 0:
                    open_ts = ts
                    open_p = high = low = close_p = price
                    vol_sum = q_take
                    dollar_sum = d_take
                else:
                    high = max(high, price)
                    low = min(low, price)
                    close_p = price
                    vol_sum += q_take
                    dollar_sum += d_take

                filled += take
                remaining -= take
                close_ts = ts

                if filled >= target_value - 1e-9:
                    flush_full_bar()

        gc.collect()

    flush_partial_if_needed()
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    return out_df.reset_index(drop=True)


class ValueBarStreamBuilder:
    """
    流式构建 volume/dollar bars：
    - 正序喂入逐笔数据
    - 在多个 zip 之间延续未完成 bar 状态
    """

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

    def _flush_partial_if_needed(self) -> None:
        if self.filled <= 0:
            return
        # 与原逻辑一致：末尾不足半个阈值则丢弃
        if self.filled < self.target_value * 0.5:
            self.filled = 0.0
            self.open_ts = None
            self.close_ts = None
            self.vol_sum = 0.0
            self.dollar_sum = 0.0
            return
        self._flush_full()

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

    def finalize(self) -> pd.DataFrame:
        self._flush_partial_if_needed()
        out = pd.DataFrame(self.rows)
        if out.empty:
            return out
        return out.reset_index(drop=True)


class DynamicThresholdPolicy:
    """
    动态阈值策略（按日）：
    1) 目标 bars/day -> 基于滚动日均成交值估算 raw threshold
    2) 对 raw threshold 做 EWMA 平滑
    3) 对相对前一日阈值施加最大变动幅度约束
    """

    def __init__(
        self,
        target_bars_per_day: int,
        rolling_days: int,
        ewma_alpha: float,
        max_daily_change: float,
        min_threshold: float | None = None,
        max_threshold: float | None = None,
    ) -> None:
        if target_bars_per_day <= 0:
            raise ValueError("target_bars_per_day 必须为正整数")
        if rolling_days <= 0:
            raise ValueError("rolling_days 必须为正整数")
        if not (0.0 < ewma_alpha <= 1.0):
            raise ValueError("ewma_alpha 必须在 (0, 1] 区间")
        if max_daily_change < 0:
            raise ValueError("max_daily_change 不能为负")

        self.target_bars_per_day = int(target_bars_per_day)
        self.rolling_days = int(rolling_days)
        self.ewma_alpha = float(ewma_alpha)
        self.max_daily_change = float(max_daily_change)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.daily_totals: list[float] = []
        self.prev_threshold: float | None = None

    def _clip_abs_bounds(self, x: float) -> float:
        y = x
        if self.min_threshold is not None:
            y = max(y, float(self.min_threshold))
        if self.max_threshold is not None:
            y = min(y, float(self.max_threshold))
        return y

    def _clip_change(self, x: float) -> float:
        if self.prev_threshold is None:
            return x
        if self.max_daily_change <= 0:
            return self.prev_threshold
        lo = self.prev_threshold * (1.0 - self.max_daily_change)
        hi = self.prev_threshold * (1.0 + self.max_daily_change)
        return min(max(x, lo), hi)

    def estimate_for_day(self, current_day_total: float) -> float:
        hist = self.daily_totals[-self.rolling_days :]
        if hist:
            mean_daily_total = float(sum(hist)) / float(len(hist))
            raw = mean_daily_total / float(self.target_bars_per_day)
        else:
            # 冷启动：避免无历史时无法估计；仅用于第一天
            raw = float(current_day_total) / float(self.target_bars_per_day)
        raw = max(raw, 1e-12)

        if self.prev_threshold is None:
            out = raw
        else:
            smoothed = (1.0 - self.ewma_alpha) * self.prev_threshold + self.ewma_alpha * raw
            out = smoothed

        out = self._clip_change(out)
        out = self._clip_abs_bounds(out)
        out = max(out, 1e-12)
        return out

    def close_day(self, realized_day_total: float, used_threshold: float) -> None:
        self.daily_totals.append(max(float(realized_day_total), 0.0))
        self.prev_threshold = max(float(used_threshold), 1e-12)


class AdaptiveValueBarStreamBuilder:
    """
    动态阈值的 volume/dollar bars 构建器：
    - 每个自然日给一个“候选阈值”
    - 若前一日遗留未完成 bar，则先沿用旧阈值把该 bar 走完，再切换到新阈值
    这样保证跨天不丢不重，且单根 bar 内阈值一致。
    """

    def __init__(self, value_col: Literal["qty", "dollar"], policy: DynamicThresholdPolicy) -> None:
        self.value_col = value_col
        self.policy = policy
        self.rows: list[dict] = []

        self.target_value = 0.0
        self.pending_target: float | None = None

        self.filled = 0.0
        self.open_ts: pd.Timestamp | None = None
        self.high = 0.0
        self.low = 0.0
        self.open_p = 0.0
        self.close_p = 0.0
        self.vol_sum = 0.0
        self.dollar_sum = 0.0
        self.close_ts: pd.Timestamp | None = None

        self.threshold_log: list[dict] = []

    def _apply_pending_target_if_possible(self) -> None:
        if self.pending_target is None:
            return
        if self.filled > 0:
            return
        self.target_value = max(float(self.pending_target), 1e-12)
        self.pending_target = None

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
        self._apply_pending_target_if_possible()

    def _flush_partial_if_needed(self) -> None:
        if self.filled <= 0:
            return
        if self.filled < self.target_value * 0.5:
            self.filled = 0.0
            self.open_ts = None
            self.close_ts = None
            self.vol_sum = 0.0
            self.dollar_sum = 0.0
            self._apply_pending_target_if_possible()
            return
        self._flush_full()

    def feed_day(self, day_df: pd.DataFrame, day_label: pd.Timestamp) -> None:
        day_total = float(day_df[self.value_col].sum())
        day_target = self.policy.estimate_for_day(day_total)
        if self.target_value <= 0:
            self.target_value = day_target
        else:
            self.pending_target = day_target
            self._apply_pending_target_if_possible()

        self.threshold_log.append(
            {
                "day": day_label,
                "value_col": self.value_col,
                "target_bars_per_day": self.policy.target_bars_per_day,
                "day_total": day_total,
                "used_target_value": self.target_value if self.pending_target is None else self.pending_target,
            }
        )

        for row in day_df.itertuples(index=False):
            ts = row.timestamp
            price = float(row.price)
            qty = float(row.qty)
            dollar = float(row.dollar)
            val_total = qty if self.value_col == "qty" else dollar
            if val_total <= 0:
                continue

            remaining = val_total
            while remaining > 1e-12:
                self._apply_pending_target_if_possible()
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

        self.policy.close_day(realized_day_total=day_total, used_threshold=day_target)

    def finalize(self) -> pd.DataFrame:
        self._flush_partial_if_needed()
        out = pd.DataFrame(self.rows)
        if out.empty:
            return out
        return out.reset_index(drop=True)

    def finalize_threshold_log(self) -> pd.DataFrame:
        out = pd.DataFrame(self.threshold_log)
        if out.empty:
            return out
        return out.sort_values("day").reset_index(drop=True)


def merge_time_bar_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """
    合并按日产生的 time bars。对同一 timestamp 的重复桶执行 OHLCV 聚合：
    - open: first
    - high: max
    - low: min
    - close: last
    - volume/dollar: sum
    """
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


def build_time_bars_from_zip_list(zip_paths: list[Path], freq: str) -> pd.DataFrame:
    """按日 zip 分别 time bar 再纵向拼接（降低峰值内存）。"""
    parts: list[pd.DataFrame] = []
    for zp in zip_paths:
        day_df = read_single_zip_trades(zp)
        if day_df.empty:
            continue
        bars = build_time_bars(day_df, freq=freq)
        if not bars.empty:
            parts.append(bars)
        del day_df
        gc.collect()
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def safe_name(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    df.to_parquet(out_path, index=False, engine="pyarrow", compression=PARQUET_COMPRESSION)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    zip_files = discover_zip_files(INPUT_DIR)
    if not zip_files:
        raise FileNotFoundError(f"未在输入目录找到 zip 文件: {INPUT_DIR}")

    if RUN_MODE not in {"incremental", "full"}:
        raise ValueError("RUN_MODE 只能是 'incremental' 或 'full'")

    auto_start = infer_start_time(OUTPUT_DIR, zip_files, AUTO_START_MODE) if RUN_MODE == "incremental" else None
    if RUN_MODE == "incremental":
        # 增量模式：显式 START_TIME 优先；否则用 auto_start（由已有输出推断）
        min_day = effective_min_zip_day(START_TIME, auto_start) if FILTER_ZIP_BY_DATE else None
    else:
        # 全量模式：只看用户显式 START_TIME，不使用 auto_start
        start_for_full = pd.Timestamp(START_TIME, tz="UTC") if START_TIME else None
        min_day = start_for_full.normalize() if (FILTER_ZIP_BY_DATE and start_for_full is not None) else None
    max_day = effective_max_zip_day(END_TIME) if FILTER_ZIP_BY_DATE else None

    zip_use = filter_zip_files(zip_files, min_day, max_day, MAX_ZIP_FILES)
    if not zip_use:
        raise RuntimeError(
            "筛选后没有可用 zip。请检查 FILTER_ZIP_BY_DATE / START_TIME / AUTO_START_MODE / END_TIME / MAX_ZIP_FILES。"
        )

    print("=== 输入概览 ===")
    print(f"RUN_MODE: {RUN_MODE}")
    print(f"目录内 zip 总数: {len(zip_files)}")
    print(f"本次参与处理 zip 数: {len(zip_use)}")
    if min_day is not None:
        print(f"zip 日期下界(文件名): {min_day.date()}")
    if max_day is not None:
        print(f"zip 日期上界(文件名): {max_day.date()}")
    if RUN_MODE == "incremental" and auto_start is not None and START_TIME is None:
        print(f"auto_start({AUTO_START_MODE}): {auto_start}")
    if MAX_ZIP_FILES is not None:
        print(f"MAX_ZIP_FILES={MAX_ZIP_FILES}（仅处理排序后的前若干个 zip）")

    # time freq 预处理（消除 pandas 的 m 别名告警）
    time_freq_pairs = [(freq, normalize_time_freq(freq)) for freq in TIME_FREQS]
    for raw, norm in time_freq_pairs:
        if raw != norm:
            print(f"[freq-normalize] {raw} -> {norm}")

    bar_types = set(BAR_TYPES)
    if VALUE_BAR_MODE not in {"fixed", "dynamic"}:
        raise ValueError("VALUE_BAR_MODE 只能是 'fixed' 或 'dynamic'")
    if VALUE_BAR_MODE == "dynamic" and (not TARGET_BARS_PER_DAY):
        raise ValueError("VALUE_BAR_MODE='dynamic' 时 TARGET_BARS_PER_DAY 不能为空")

    # ------------------------------------------------------------
    # 单次正序遍历 zip：同步更新三类 bars（避免重复读取 zip）
    # ------------------------------------------------------------
    time_parts: dict[str, list[pd.DataFrame]] = {freq: [] for freq in TIME_FREQS} if "time" in bar_types else {}
    if VALUE_BAR_MODE == "fixed":
        vol_builders = (
            {float(v): ValueBarStreamBuilder(float(v), "qty") for v in VOLUME_THRESHOLDS}
            if "volume" in bar_types
            else {}
        )
        dol_builders = (
            {float(d): ValueBarStreamBuilder(float(d), "dollar") for d in DOLLAR_THRESHOLDS}
            if "dollar" in bar_types
            else {}
        )
        vol_dyn_builders: dict[int, AdaptiveValueBarStreamBuilder] = {}
        dol_dyn_builders: dict[int, AdaptiveValueBarStreamBuilder] = {}
    else:
        vol_builders = {}
        dol_builders = {}
        vol_dyn_builders = (
            {
                int(k): AdaptiveValueBarStreamBuilder(
                    "qty",
                    DynamicThresholdPolicy(
                        target_bars_per_day=int(k),
                        rolling_days=DYNAMIC_ROLLING_DAYS,
                        ewma_alpha=DYNAMIC_EWMA_ALPHA,
                        max_daily_change=DYNAMIC_MAX_DAILY_CHANGE,
                        min_threshold=DYNAMIC_MIN_THRESHOLD,
                        max_threshold=DYNAMIC_MAX_THRESHOLD,
                    ),
                )
                for k in TARGET_BARS_PER_DAY
            }
            if "volume" in bar_types
            else {}
        )
        dol_dyn_builders = (
            {
                int(k): AdaptiveValueBarStreamBuilder(
                    "dollar",
                    DynamicThresholdPolicy(
                        target_bars_per_day=int(k),
                        rolling_days=DYNAMIC_ROLLING_DAYS,
                        ewma_alpha=DYNAMIC_EWMA_ALPHA,
                        max_daily_change=DYNAMIC_MAX_DAILY_CHANGE,
                        min_threshold=DYNAMIC_MIN_THRESHOLD,
                        max_threshold=DYNAMIC_MAX_THRESHOLD,
                    ),
                )
                for k in TARGET_BARS_PER_DAY
            }
            if "dollar" in bar_types
            else {}
        )

    n_zip = len(zip_use)
    t0 = time.time()
    trades_so_far = 0
    t_min: pd.Timestamp | None = None
    t_max: pd.Timestamp | None = None
    for i, zp in enumerate(zip_use, start=1):
        day_df = read_single_zip_trades(zp)
        if day_df.empty:
            continue
        trades_so_far += len(day_df)
        dmin = day_df["timestamp"].min()
        dmax = day_df["timestamp"].max()
        if t_min is None or dmin < t_min:
            t_min = dmin
        if t_max is None or dmax > t_max:
            t_max = dmax

        if "time" in bar_types:
            for raw_freq, norm_freq in time_freq_pairs:
                day_bars = build_time_bars(day_df, freq=norm_freq)
                if not day_bars.empty:
                    time_parts[raw_freq].append(day_bars)

        if "volume" in bar_types:
            if VALUE_BAR_MODE == "fixed":
                for b in vol_builders.values():
                    b.feed_df(day_df)
            else:
                day_label = day_df["timestamp"].iloc[0].normalize()
                for b in vol_dyn_builders.values():
                    b.feed_day(day_df, day_label=day_label)

        if "dollar" in bar_types:
            if VALUE_BAR_MODE == "fixed":
                for b in dol_builders.values():
                    b.feed_df(day_df)
            else:
                day_label = day_df["timestamp"].iloc[0].normalize()
                for b in dol_dyn_builders.values():
                    b.feed_day(day_df, day_label=day_label)

        if (PROGRESS_EVERY_ZIP <= 1) or (i % PROGRESS_EVERY_ZIP == 0) or (i == n_zip):
            pct = i / n_zip if n_zip > 0 else 1.0
            bar_w = 30
            filled = int(bar_w * pct)
            prog_bar = "[" + "#" * filled + "-" * (bar_w - filled) + "]"
            elapsed = time.time() - t0
            msg = (
                f"[progress] {prog_bar} {pct:.1%} "
                f"zip {i}/{n_zip} | trades_so_far={trades_so_far:,} | elapsed={elapsed/60:.1f} min"
            )
            if i < n_zip:
                print(msg, end="\r", flush=True)
            else:
                print(msg, flush=True)

        del day_df
        gc.collect()

    # time 输出
    if "time" in bar_types:
        for freq in TIME_FREQS:
            bars = merge_time_bar_chunks(time_parts[freq])
            out = OUTPUT_DIR / f"time_{freq}.parquet"
            write_parquet(bars, out)
            print(f"[time] {freq} -> {out} rows={len(bars)}")

    # volume 输出
    if "volume" in bar_types:
        if VALUE_BAR_MODE == "fixed":
            for v, builder in vol_builders.items():
                bars = builder.finalize()
                out = OUTPUT_DIR / f"volume_{safe_name(float(v))}.parquet"
                write_parquet(bars, out)
                print(f"[volume] {v} -> {out} rows={len(bars)}")
        else:
            for k, builder in vol_dyn_builders.items():
                bars = builder.finalize()
                out = OUTPUT_DIR / f"volume_dynamic_k{k}.parquet"
                write_parquet(bars, out)
                print(f"[volume-dynamic] k={k} -> {out} rows={len(bars)}")
                th = builder.finalize_threshold_log()
                th_out = OUTPUT_DIR / f"volume_dynamic_k{k}_thresholds.parquet"
                write_parquet(th, th_out)
                print(f"[volume-dynamic] k={k} threshold-log -> {th_out} rows={len(th)}")

    # dollar 输出
    if "dollar" in bar_types:
        if VALUE_BAR_MODE == "fixed":
            for d, builder in dol_builders.items():
                bars = builder.finalize()
                out = OUTPUT_DIR / f"dollar_{safe_name(float(d))}.parquet"
                write_parquet(bars, out)
                print(f"[dollar] {d} -> {out} rows={len(bars)}")
        else:
            for k, builder in dol_dyn_builders.items():
                bars = builder.finalize()
                out = OUTPUT_DIR / f"dollar_dynamic_k{k}.parquet"
                write_parquet(bars, out)
                print(f"[dollar-dynamic] k={k} -> {out} rows={len(bars)}")
                th = builder.finalize_threshold_log()
                th_out = OUTPUT_DIR / f"dollar_dynamic_k{k}_thresholds.parquet"
                write_parquet(th, th_out)
                print(f"[dollar-dynamic] k={k} threshold-log -> {th_out} rows={len(th)}")

    meta = pd.DataFrame(
        {
            "input_dir": [str(INPUT_DIR)],
            "output_dir": [str(OUTPUT_DIR)],
            "bar_types": [",".join(BAR_TYPES)],
            "value_bar_mode": [VALUE_BAR_MODE],
            "time_freqs": [",".join(TIME_FREQS)],
            "volume_thresholds": [",".join(str(x) for x in VOLUME_THRESHOLDS)],
            "dollar_thresholds": [",".join(str(x) for x in DOLLAR_THRESHOLDS)],
            "target_bars_per_day": [",".join(str(int(x)) for x in TARGET_BARS_PER_DAY)],
            "dynamic_rolling_days": [DYNAMIC_ROLLING_DAYS],
            "dynamic_ewma_alpha": [DYNAMIC_EWMA_ALPHA],
            "dynamic_max_daily_change": [DYNAMIC_MAX_DAILY_CHANGE],
            "dynamic_min_threshold": [DYNAMIC_MIN_THRESHOLD if DYNAMIC_MIN_THRESHOLD is not None else ""],
            "dynamic_max_threshold": [DYNAMIC_MAX_THRESHOLD if DYNAMIC_MAX_THRESHOLD is not None else ""],
            "start_time": [START_TIME or (str(auto_start) if auto_start is not None else "")],
            "end_time": [END_TIME or ""],
            "min_ts": [str(t_min) if t_min is not None else ""],
            "max_ts": [str(t_max) if t_max is not None else ""],
            "n_zips_used": [len(zip_use)],
            "stream_volume_dollar": [STREAM_VOLUME_DOLLAR],
        }
    )
    write_parquet(meta, OUTPUT_DIR / "build_meta.parquet")
    print(f"meta -> {OUTPUT_DIR / 'build_meta.parquet'}")


if __name__ == "__main__":
    main()

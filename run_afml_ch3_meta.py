import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 复用你已经验证跑通的第 3 章实现（不改 class3.py）
from run_afml_ch3 import (
    addVerticalBarrier,
    dropLabels,
    getBins,
    getDailyVol,
    getEvents,
    getTEvents,
)


@dataclass(frozen=True)
class MetaConfig:
    # === 输入数据 ===
    dollar_bars_parquet: str = "bars_parquet/BTCUSDT_2021-01-01_2021-01-10_dollar_bars.parquet"
    ts_col: str = "timestamp"
    close_col: str = "close"

    # === 3.1 每日波动率估算 ===
    daily_vol_span: int = 100

    # === 3.2 CUSUM 事件过滤器阈值 ===
    cusum_h_mode: str = "last"  # "last" | "mean"

    # === 3.4 垂直屏障 ===
    num_days: int = 1

    # === 3.2/3.3 三重屏障 ===
    pt_sl: tuple[float, float] = (2.0, 1.0)
    min_ret: float = 1e-4
    num_threads: int = 1

    # === Meta-labeling: 方向信号 side（行业常见：用简单规则信号作为“第一阶段”）===
    # 用快慢均线方向：fast_ma > slow_ma -> +1，否则 -1
    side_fast_window: int = 5
    side_slow_window: int = 20

    # === 输出 ===
    out_dir: str = "outputs/ch3_meta"


def make_side_from_ma(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    生成 side（方向信号）：
    - fast_sma > slow_sma => +1
    - fast_sma <= slow_sma => -1
    说明：这是最简行业基线信号，用于演示 meta-labeling 流程。
    """
    fast_ma = close.rolling(window=fast, min_periods=fast).mean()
    slow_ma = close.rolling(window=slow, min_periods=slow).mean()
    side = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=close.index, dtype="float64")
    side = side.where(fast_ma.notna() & slow_ma.notna())
    return side


def main() -> None:
    cfg = MetaConfig()

    df = pd.read_parquet(cfg.dollar_bars_parquet)
    if cfg.ts_col in df.columns:
        df[cfg.ts_col] = pd.to_datetime(df[cfg.ts_col])
        df = df.set_index(cfg.ts_col)

    close = pd.to_numeric(df[cfg.close_col], errors="raise")
    close.index = pd.to_datetime(close.index).tz_localize(None)

    # 3.1：日波动率（并对齐可用区间）
    daily_vol = getDailyVol(close, span0=cfg.daily_vol_span)
    close_ = close.loc[daily_vol.index]

    if cfg.cusum_h_mode == "mean":
        h = float(daily_vol.mean())
    else:
        h = float(daily_vol.dropna().iloc[-1])

    # 3.2：CUSUM events（用 log price，与你 notebook 的写法一致）
    tEvents = getTEvents(np.log(close_), h=h)

    # 3.4：垂直屏障
    t1 = addVerticalBarrier(tEvents, close_, numDays=cfg.num_days)

    # Meta：生成方向 side，并对齐到事件时刻
    side_all = make_side_from_ma(close_, fast=cfg.side_fast_window, slow=cfg.side_slow_window)
    side = side_all.reindex(tEvents)
    side = side.dropna()

    # 仅保留有 side 的事件（否则 meta-labeling 没意义）
    tEvents_side = pd.DatetimeIndex(side.index)
    t1_side = t1.reindex(tEvents_side)

    # 3.3：带 side 的 events（meta-labeling）
    events = getEvents(
        close=close_,
        tEvents=tEvents_side,
        ptSl=cfg.pt_sl,
        trgt=daily_vol,
        minRet=cfg.min_ret,
        numThreads=cfg.num_threads,
        t1=t1_side,
        side=side,
    )

    # 3.7：meta bins（当提供 side 时：不赚钱 => bin=0；赚钱 => bin=1）
    bins = getBins(events, close_)
    bins_dropped = dropLabels(bins, minPct=0.05)

    os.makedirs(cfg.out_dir, exist_ok=True)
    events_path = os.path.join(cfg.out_dir, "events_meta.parquet")
    bins_path = os.path.join(cfg.out_dir, "bins_meta.parquet")
    bins_dropped_path = os.path.join(cfg.out_dir, "bins_meta_dropped.parquet")
    side_path = os.path.join(cfg.out_dir, "side.parquet")

    events.to_parquet(events_path)
    bins.to_parquet(bins_path)
    bins_dropped.to_parquet(bins_dropped_path)
    side.to_frame("side").to_parquet(side_path)

    # 一些直观指标（帮助理解 meta-labeling 在“过滤交易”上的作用）
    exec_rate = float((bins["bin"] == 1).mean()) if len(bins) else float("nan")
    avg_ret_all = float(bins["ret"].mean()) if len(bins) else float("nan")
    avg_ret_exec = float(bins.loc[bins["bin"] == 1, "ret"].mean()) if (bins["bin"] == 1).any() else float("nan")

    print(f"close bars (full): {len(close)}")
    print(f"close bars (aligned to daily_vol): {len(close_)}")
    print(f"daily_vol: {int(daily_vol.notna().sum())} (non-NaN)")
    print(f"cusum h={h:.8f} tEvents(all)={len(tEvents)}")
    print(f"side available on events={len(tEvents_side)} (fast={cfg.side_fast_window}, slow={cfg.side_slow_window})")
    print(f"vertical barriers t1_side={int(t1_side.notna().sum())}")
    print(f"events(meta, after minRet filter)={len(events)}")
    print("bins(meta) distribution:\n", bins["bin"].value_counts(dropna=False))
    print(f"exec_rate (bin==1)={exec_rate:.3f}")
    print(f"avg_ret_all (ret already * side)={avg_ret_all:.6f}")
    print(f"avg_ret_exec (only bin==1)={avg_ret_exec:.6f}")
    print(f"wrote: {events_path}")
    print(f"wrote: {bins_path}")
    print(f"wrote: {bins_dropped_path}")
    print(f"wrote: {side_path}")


if __name__ == "__main__":
    main()


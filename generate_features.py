import os
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from IPython.display import display




def getDailyVol(close: pd.Series, span0: int = 100, days: int = 1) -> pd.Series:
    # 取“当前时刻 vs days 前最近可对齐的时刻”的对数收益率，并用 EWMA 标准差估计波动率
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=days))
    df0 = df0[df0 > 0]

    prev_index = close.index[df0 - 1]  # days 前的对齐时间戳
    curr_index = close.index[close.shape[0] - df0.shape[0] :]  # 当前时间戳（与 prev_index 等长对齐）

    log_ret = np.log(close.loc[curr_index].values / close.loc[prev_index].values)
    log_ret = pd.Series(log_ret, index=curr_index)

    return log_ret.ewm(span=span0).std()


def getTEvents(gRaw: pd.Series, h: float) -> pd.DatetimeIndex:
    tEvents: list[pd.Timestamp] = []
    sPos, sNeg = 0.0, 0.0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0.0, sPos + float(diff.loc[i])), min(0.0, sNeg + float(diff.loc[i]))
        if sNeg < -h:
            sNeg = 0.0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0.0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def addVerticalBarrier(tEvents: pd.DatetimeIndex, close: pd.Series, numDays: int = 1) -> pd.Series:
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    return pd.Series(close.index[t1], index=tEvents[: t1.shape[0]])


def mpPandasObj(
    func: Callable[..., pd.DataFrame],
    pdObj: tuple[str, pd.Index],
    numThreads: int,
    **kargs,
) -> pd.DataFrame:
    """
    书里用 multiprocessing 把 index 切片后并行跑。
    为了“调用逻辑一致 + 可跑通”，这里提供兼容接口：
    - numThreads<=1 时串行一次跑完
    - numThreads>1 时仍然串行分块（不引入多进程依赖/序列化问题）
    """
    name, index = pdObj
    if numThreads is None or int(numThreads) <= 1:
        return func(molecule=index, **kargs)

    n = int(numThreads)
    parts: list[pd.Index] = []
    step = int(np.ceil(len(index) / n))
    for i in range(0, len(index), step):
        parts.append(index[i : i + step])

    out = []
    for part in parts:
        out.append(func(molecule=part, **kargs))
    return pd.concat(out).sort_index()


def applyPtSlOnT1(close: pd.Series, events: pd.DataFrame, ptSl: tuple[float, float]) -> pd.DataFrame:
    out = events[["t1"]].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events["trgt"]
    else:
        pt = pd.Series(index=events.index, dtype="float64")

    if ptSl[1] > 0:
        sl = -ptSl[1] * events["trgt"]
    else:
        sl = pd.Series(index=events.index, dtype="float64")

    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        df0 = close.loc[loc:t1]
        df0 = (df0 / close.loc[loc] - 1.0) * events.at[loc, "side"]
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()

    return out



# def getEvents(close: pd.Series,tEvents: pd.DatetimeIndex,ptSl: tuple[float, float],trgt: pd.Series,minRet: float,t1: pd.Series | bool = False,side: pd.Series | None = None,) -> pd.DataFrame:
#     trgt = trgt.loc[tEvents]
#     trgt = trgt[trgt > minRet]
#     if t1 is False:
#         t1 = pd.Series(pd.NaT, index=tEvents)
#     if side is None:
#         side_, ptSl_ = pd.Series(1.0, index=trgt.index), (ptSl[0], ptSl[0])
#     else:
#         side_, ptSl_ = side.loc[trgt.index], ptSl
#     events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])
#     df0 = applyPtSlOnT1(molecule=events.index,close=close,events=events,ptSl=ptSl_)
#     events["t1"] = df0.dropna(how="all").min(axis=1)
#     if side is None:
#         events = events.drop(columns=["side"])
#     return events


def getEvents(
    close: pd.Series,
    start_time: pd.DatetimeIndex,
    ptSl: tuple[float, float],
    volatility: pd.Series,
    minRet: float,
    start_end_time: pd.Series | bool = False,
    side: pd.Series | None = None,
) -> pd.DataFrame:

    volatility = volatility.loc[start_time]
    volatility = volatility[volatility > minRet]

    if start_end_time is False:
        start_end_time = pd.Series(pd.NaT, index=start_time)

    if side is None:
        side_, ptSl_ = pd.Series(1.0, index=volatility.index), (ptSl[0], ptSl[0])
    else:
        side_, ptSl_ = side.loc[volatility.index], ptSl

    events = pd.concat({"t1": start_end_time, "trgt": volatility, "side": side_}, axis=1).dropna(subset=["trgt"])
    df0 = applyPtSlOnT1(close=close,events=events,ptSl=ptSl_)

    events["t1"] = df0.dropna(how="all").min(axis=1)

    if side is None:
        events = events.drop(columns=["side"])

    return events


def getBins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")

    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index].values - 1.0

    if "side" in events_.columns:
        out["ret"] *= events_["side"]

    out["bin"] = np.sign(out["ret"])

    if "side" in events_.columns:
        out.loc[out["ret"] <= 0, "bin"] = 0

    return out


def dropLabels(bins: pd.DataFrame, minPct: float = 0.05) -> pd.DataFrame:
    out = bins.copy()
    while True:
        df0 = out["bin"].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        dropped = df0.idxmin()
        out = out[out["bin"] != dropped]
    return out




@dataclass(frozen=True)
class Config:
    # === 输入数据（和你 notebook 一致：dollar bars parquet）===
    dollar_bars_parquet: str = "bars_parquet/BTCUSDT_2021-01-01_2021-01-10_dollar_bars.parquet"


    # === 3.1 每日波动率估算 ===
    daily_vol_span: int = 100

    # === 3.2/3.3 CUSUM 事件过滤器阈值 ===
    # 书里常见做法：h 取日波动率序列某种统计量。这里默认取最后一个有效值（更贴近你 notebook）。
    cusum_h_mode: str = "mean"  # "last" | "mean"

    # === 3.4 垂直屏障 ===
    num_days: int = 1

    # === 3.2 三重屏障（pt/sl 宽度为波动率倍数）===
    pt_sl: tuple[float, float] = (2.0, 1.0)
    min_ret: float = 1e-4
    num_threads: int = 1  # 书里是并行，这里提供兼容接口；默认单线程可跑通

    # === 输出 ===
    out_dir: str = "outputs/features"


def main() -> None:
    cfg = Config()
    df = pd.read_parquet(cfg.dollar_bars_parquet)


    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').tz_localize(None)
    close = pd.to_numeric(df['close'], errors="raise")

    # 注意：书里的 getDailyVol 会丢掉最早那一段（无法回溯 1 day 的部分）
    # 为了保持 getEvents(trgt.loc[tEvents]) 这一“书中调用逻辑”可用，
    # 我们在生成 tEvents/vertical barrier/events 时，只在 daily_vol 的有效索引区间内工作。
    daily_vol = getDailyVol(close, span0=cfg.daily_vol_span)
    close_ = close.loc[daily_vol.index]
    print('波动率统计分析',daily_vol.describe())

    if cfg.cusum_h_mode == "mean":
        h = float(daily_vol.mean())
    else:
        h = float(daily_vol.dropna().iloc[-1])

    # 书里通常对 log price（或 log return 累积）应用 CUSUM；你 notebook 是把 log(price) 直接喂给 getTEvents
    gRaw = np.log(close_)
    t_start = getTEvents(gRaw, h=h)
    print(t_start.shape)
    print(t_start[:1])

    # 垂直屏障：只对能找到未来 numDays 的事件保留 t1
    t_start_end = addVerticalBarrier(t_start, close_, numDays=cfg.num_days)
    print(t_start_end.shape)
    print(t_start_end.head(3))

    # 目标波动率 trgt：书里用 dailyVol，在 tEvents 上对齐后再过滤 minRet
    events = getEvents(close=close_,start_time=t_start,ptSl=cfg.pt_sl,volatility=daily_vol,minRet=cfg.min_ret,start_end_time=t_start_end,side=None)
    print(events.shape)
    print(events.head(3))

    bins = getBins(events, close_)
    print(bins)
    bins_dropped = dropLabels(bins, minPct=0.05)

    # os.makedirs(cfg.out_dir, exist_ok=True)
    # events_path = os.path.join(cfg.out_dir, "events.parquet")
    # bins_path = os.path.join(cfg.out_dir, "bins.parquet")
    # bins_dropped_path = os.path.join(cfg.out_dir, "bins_dropped.parquet")
    # events.to_parquet(events_path)
    # bins.to_parquet(bins_path)
    # bins_dropped.to_parquet(bins_dropped_path)

    # 控制台关键统计（对照书里：样本数量、标签分布）
    print(f"close bars (full): {len(close)}")
    print(f"close bars (aligned to daily_vol): {len(close_)}")
    print(f"daily_vol: {len(daily_vol.dropna())} (non-NaN)")
    print(f"cusum h={h:.8f} tEvents={len(t_start)}")
    print(f"vertical barriers t1={len(t_start_end)}")
    print(f"events(after minRet filter)={len(events)}")
    print("bins distribution:\n", bins["bin"].value_counts(dropna=False))
    print("bins_dropped distribution:\n", bins_dropped["bin"].value_counts(dropna=False))



if __name__ == "__main__":
    main()


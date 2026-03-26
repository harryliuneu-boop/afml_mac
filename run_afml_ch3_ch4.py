import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


# =========================
# Chapter 3 (from run_afml_ch3.py)
# =========================


@dataclass(frozen=True)
class PipelineConfig:
    # === 输入数据 ===
    dollar_bars_parquet: str = "bars_parquet/BTCUSDT_2021-01-01_2021-01-10_dollar_bars.parquet"
    ts_col: str = "timestamp"
    close_col: str = "close"

    # === 3.1 每日波动率估算 ===
    daily_vol_span: int = 100

    # === 3.2/3.3 CUSUM 事件过滤器阈值 ===
    # 书中常见做法：h 取 dailyVol 的统计量（例如均值），这样事件不会过密，后续三重屏障更易跑通
    cusum_h_mode: str = "mean"  # "last" | "mean"

    # === 3.4 垂直屏障 ===
    num_days: int = 1

    # === 3.2 三重屏障（pt/sl 宽度为波动率倍数）===
    pt_sl: tuple[float, float] = (2.0, 1.0)
    min_ret: float = 1e-4
    num_threads: int = 1

    # === 输出 ===
    out_dir: str = "outputs/ch3_ch4"


def getDailyVol(close: pd.Series, span0: int = 100) -> pd.Series:
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0


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
    func: Callable[..., pd.DataFrame | pd.Series],
    pdObj: tuple[str, pd.Index],
    numThreads: int,
    **kargs,
) -> pd.DataFrame | pd.Series:
    """
    书里用 multiprocessing 把 index 切片后并行跑。
    为了“调用逻辑一致 + 可跑通”，这里提供兼容接口：
    - numThreads<=1 时串行一次跑完
    - numThreads>1 时仍然串行分块（不引入多进程依赖/序列化问题）
    同时兼容 func 返回 DataFrame / Series。
    """
    _, index = pdObj
    if numThreads is None or int(numThreads) <= 1:
        return func(molecule=index, **kargs)

    n = int(numThreads)
    parts: list[pd.Index] = []
    step = int(np.ceil(len(index) / n))
    for i in range(0, len(index), step):
        parts.append(index[i : i + step])

    outs: list[pd.DataFrame | pd.Series] = []
    for part in parts:
        outs.append(func(molecule=part, **kargs))

    if not outs:
        return pd.DataFrame()
    if isinstance(outs[0], pd.Series):
        return pd.concat(outs).sort_index()
    return pd.concat(outs).sort_index()


def applyPtSlOnT1(close: pd.Series, events: pd.DataFrame, ptSl: tuple[float, float], molecule: Iterable) -> pd.DataFrame:
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events.index, dtype="float64")

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events.index, dtype="float64")

    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        df0 = close.loc[loc:t1]
        df0 = (df0 / close.loc[loc] - 1.0) * events_.at[loc, "side"]
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()

    return out


def getEvents(
    close: pd.Series,
    tEvents: pd.DatetimeIndex,
    ptSl: tuple[float, float],
    trgt: pd.Series,
    minRet: float,
    numThreads: int,
    t1: pd.Series | bool = False,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]

    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    if side is None:
        side_, ptSl_ = pd.Series(1.0, index=trgt.index), (ptSl[0], ptSl[0])
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl

    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])

    df0 = mpPandasObj(
        func=applyPtSlOnT1,
        pdObj=("molecule", events.index),
        numThreads=numThreads,
        close=close,
        events=events,
        ptSl=ptSl_,
    )
    assert isinstance(df0, pd.DataFrame)
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


# =========================
# Chapter 4 (from class4.py, re-stitched to be runnable)
# =========================


def mpNumCoEvents(closeIdx: pd.Index, t1: pd.Series, molecule: Sequence[pd.Timestamp]) -> pd.Series:
    """
    class4.py 的 4.1：计算每一根 bar 上的并发事件数量（并只返回 molecule 覆盖范围）
    """
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[: t1[molecule].max()]

    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0] : iloc[1] + 1])
    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1
    return count.loc[molecule[0] : t1[molecule].max()]


def mpSampleTW(t1: pd.Series, numCoEvents: pd.Series, molecule: Sequence[pd.Timestamp]) -> pd.Series:
    """
    class4.py 的 4.2：对一个标签的平均唯一性进行估算（Time-Weighted Uniqueness）
    """
    wght = pd.Series(index=pd.Index(molecule), dtype="float64")
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1.0 / numCoEvents.loc[tIn:tOut]).mean()
    return wght


def getIndMatrix(barIx: Sequence, t1: pd.Series) -> pd.DataFrame:
    indM = pd.DataFrame(0.0, index=pd.Index(barIx), columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.items()):
        indM.loc[t0:t1_, i] = 1.0
    return indM


def getAvgUniqueness(indM: pd.DataFrame) -> pd.Series:
    c = indM.sum(axis=1)
    u = indM.div(c, axis=0)
    avgU = u[u > 0].mean()
    return avgU


def seqBootstrap(indM: pd.DataFrame, sLength: int | None = None) -> list[int]:
    if sLength is None:
        sLength = indM.shape[1]
    phi: list[int] = []
    while len(phi) < sLength:
        avgU = pd.Series(dtype="float64")
        for i in indM:
            indM_ = indM[phi + [i]]
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()
        phi += [int(np.random.choice(indM.columns, p=prob))]
    return phi


def mpSampleWRet(t1: pd.Series, numCoEvents: pd.Series, close: pd.Series, molecule: Sequence[pd.Timestamp]) -> pd.Series:
    """
    class4.py 的 4.10：绝对回报归因法样本权重
    """
    ret = np.log(close).diff()
    wght = pd.Series(index=pd.Index(molecule), dtype="float64")
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def getTimeDecay(tW: pd.Series, clfLastW: float = 1.0) -> pd.Series:
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    return clfW


def main() -> None:
    cfg = PipelineConfig()

    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_parquet(cfg.dollar_bars_parquet)
    if cfg.ts_col in df.columns:
        df[cfg.ts_col] = pd.to_datetime(df[cfg.ts_col])
        df = df.set_index(cfg.ts_col)

    close = pd.to_numeric(df[cfg.close_col], errors="raise")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    print(f"Loaded bars: {len(close)}", flush=True)

    # ---- Chapter 3: Events + Labels ----
    daily_vol = getDailyVol(close, span0=cfg.daily_vol_span)
    close_ = close.loc[daily_vol.index]
    print(f"Daily vol computed: non-NaN={daily_vol.notna().sum()} aligned_close={len(close_)}", flush=True)

    h = float(daily_vol.mean()) if cfg.cusum_h_mode == "mean" else float(daily_vol.dropna().iloc[-1])
    gRaw = np.log(close_)
    tEvents = getTEvents(gRaw, h=h)
    print(f"CUSUM done: h={h:.8f} tEvents={len(tEvents)}", flush=True)
    t1 = addVerticalBarrier(tEvents, close_, numDays=cfg.num_days)
    print(f"Vertical barrier done: t1={len(t1)}", flush=True)

    events = getEvents(close=close_,tEvents=tEvents,ptSl=cfg.pt_sl,trgt=daily_vol,minRet=cfg.min_ret,numThreads=cfg.num_threads,t1=t1,side=None)
    print(f"Triple barrier done: events={len(events)}", flush=True)
    bins = getBins(events, close_)
    bins_dropped = dropLabels(bins, minPct=0.05)
    print("Labeling done.", flush=True)


    # ---- Chapter 4: Uniqueness + Sequential Bootstrap + Sample Weights ----
    # 4.1 并发事件数
    numCoEvents = mpPandasObj(mpNumCoEvents,("molecule", events.index),cfg.num_threads,closeIdx=close_.index,t1=events["t1"])
    assert isinstance(numCoEvents, pd.Series)
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep="last")]
    numCoEvents = numCoEvents.reindex(close_.index).fillna(0.0)

    # 4.2 平均唯一性（tW）
    tW = mpPandasObj(mpSampleTW,("molecule", events.index),cfg.num_threads,t1=events["t1"],numCoEvents=numCoEvents)
    assert isinstance(tW, pd.Series)

    # 4.3-4.5 指示矩阵 + 序列自助法
    indM = getIndMatrix(close_.index, events["t1"])
    phi_seq = seqBootstrap(indM, sLength=indM.shape[1])

    # 4.10 样本权重（绝对回报归因）并归一化
    w = mpPandasObj(mpSampleWRet,("molecule", events.index),cfg.num_threads,t1=events["t1"],numCoEvents=numCoEvents,close=close_)
    assert isinstance(w, pd.Series)
    w = w * (w.shape[0] / w.sum())

    # 时间衰减（示例：最后一个样本权重=0.5）
    w_decay = getTimeDecay(tW, clfLastW=0.5)

    # ---- Save & Print ----
    os.makedirs(cfg.out_dir, exist_ok=True)
    events.to_parquet(os.path.join(cfg.out_dir, "events.parquet"))
    bins.to_parquet(os.path.join(cfg.out_dir, "bins.parquet"))
    bins_dropped.to_parquet(os.path.join(cfg.out_dir, "bins_dropped.parquet"))
    tW.to_frame("tW").to_parquet(os.path.join(cfg.out_dir, "tW.parquet"))
    w.to_frame("w").to_parquet(os.path.join(cfg.out_dir, "w.parquet"))
    w_decay.to_frame("w_decay").to_parquet(os.path.join(cfg.out_dir, "w_decay.parquet"))

    print("=== Chapter 3 ===")
    print(f"close bars (full): {len(close)}")
    print(f"close bars (aligned to daily_vol): {len(close_)}")
    print(f"daily_vol (non-NaN): {len(daily_vol.dropna())}")
    print(f"cusum h={h:.8f} tEvents={len(tEvents)} vertical t1={len(t1)} events={len(events)}")
    print("bins distribution:\n", bins["bin"].value_counts(dropna=False))
    print("bins_dropped distribution:\n", bins_dropped["bin"].value_counts(dropna=False))

    print("\n=== Chapter 4 ===")
    print(f"numCoEvents (nonzero bars): {(numCoEvents > 0).sum()} / {len(numCoEvents)}")
    print("tW summary:\n", tW.describe())
    print("w summary:\n", w.describe())
    print("w_decay summary:\n", w_decay.describe())
    print(f"sequential bootstrap phi (first 10): {phi_seq[:10]}")


if __name__ == "__main__":
    main()


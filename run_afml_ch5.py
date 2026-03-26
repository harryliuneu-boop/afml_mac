import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 复用第 3 章的数据配置，保证和前面章节用的是同一份 BTCUSDT 数据
from run_afml_ch3_ch4 import PipelineConfig


# =========================
# 5.1 加权函数（分数阶微分系数）
# =========================


def getWeights(d: float, size: int) -> np.ndarray:
    """
    标准分数阶微分权重（扩展窗口），对应书中 5.1 与 class5.py:getWeights。
    返回形状为 (size, 1) 的列向量，靠近序列末尾的权重对应较新的观测。
    """
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plotWeights(dRange: Sequence[float], nPlots: int, size: int) -> None:
    """
    画出不同 d 下的权重形状，直观理解“记忆衰减速度”。
    """
    w_df = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[round(float(d), 4)])
        w_df = w_df.join(w_, how="outer")
    ax = w_df.plot()
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# =========================
# 5.2 标准分数阶微分（扩展窗口）
# =========================


def fracDiff(series: pd.DataFrame, d: float, thres: float = 0.01) -> pd.DataFrame:
    """
    使用扩展窗口计算分数阶微分，基本等价于 class5.py:fracDiff。

    参数:
    - series: 多列价格数据（例如 log 价格）
    - d: 分数阶阶数
    - thres: 累计权重阈值，用于裁剪前端很小的权重，以减少计算量
    """
    # 1) 为最长序列计算权重
    w = getWeights(d, series.shape[0])

    # 2) 根据权重损失阈值确定需要跳过的初始位置
    w_ = np.cumsum(np.abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]

    # 3) 对每一列应用权重
    out: dict[str, pd.Series] = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype="float64")

        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue
            df_[loc] = float(np.dot(w[-(iloc + 1) :, :].T, seriesF.iloc[: iloc + 1])[0, 0])

        out[name] = df_.copy(deep=True)

    df = pd.concat(out, axis=1)
    return df


# =========================
# 5.3 定宽窗口分数阶微分（FFD）
# =========================


def getWeights_FFD(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    为定宽窗口 FFD 生成权重，直到权重绝对值小于 thres 为止。
    对应书中 5.3 的权重截断思想。
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff_FFD(series: pd.DataFrame, d: float, thres: float = 1e-5) -> pd.DataFrame:
    """
    Fast Fractional Differencing (定宽窗口分数阶微分)，对应 class5.py:fracDiff_FFD。
    """
    # 1) 为最长序列计算固定权重的窗口
    w = getWeights_FFD(d, thres=thres)
    width = len(w) - 1

    # 2) 应用到每一列
    out: dict[str, pd.Series] = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype="float64")

        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = float(np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0])

        out[name] = df_.copy(deep=True)

    df = pd.concat(out, axis=1)
    return df


# =========================
# 5.4 寻找通过 ADF 测试的最小 d （改写成直接用第 3 章 BTC 数据）
# =========================


@dataclass(frozen=True)
class Ch5Config:
    # 使用前面章节的 BTCUSDT close 作为示例
    data_cfg: PipelineConfig = PipelineConfig()
    out_dir: str = "outputs/ch5"
    inst_name: str = "BTCUSDT_FFD"


def load_close_from_ch3(cfg: PipelineConfig) -> pd.Series:
    """
    读取与第 3 章相同的数据源，返回 close 序列。
    """
    df = pd.read_parquet(cfg.dollar_bars_parquet)
    if cfg.ts_col in df.columns:
        df[cfg.ts_col] = pd.to_datetime(df[cfg.ts_col])
        df = df.set_index(cfg.ts_col)
    close = pd.to_numeric(df[cfg.close_col], errors="raise")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def plotMinFFD_from_series(price: pd.Series, cfg: Ch5Config) -> None:
    """
    5.4 的逻辑：在一系列 d 上做 FFD，比较 ADF 统计量和平稳性与原序列的相关性。
    这里直接使用传入的 price（例如 BTCUSDT 收盘价），不依赖特定 CSV 文件。
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 存储 ADF+相关性的统计结果
    out = pd.DataFrame(columns=["adfStat", "pVal", "lags", "nObs", "95% conf", "corr"])

    # 1. 对数 + 重采样为日线（减少微观噪声）
    # df1 = np.log(price).to_frame("Close").resample("1D").last().dropna()
    df1 = np.log(price).to_frame("Close").dropna()
    for d in np.linspace(0, 1, 11):
        # 2. FFD
        df2 = fracDiff_FFD(df1, d, thres=0.01)

        # 若太短，跳过（ADF 需要足够点数）
        if df2["Close"].dropna().shape[0] < 20:
            continue

        # 3. 与原序列的相关性（记忆保留程度）
        aligned_idx = df2.index.intersection(df1.index)
        corr = float(
            np.corrcoef(df1.loc[aligned_idx, "Close"], df2.loc[aligned_idx, "Close"])[0, 1]
        )

        # 4. ADF 平稳性检验
        adf_res = adfuller(df2["Close"].dropna(), maxlag=1, regression="c", autolag=None)

        # 5. 存储结果
        out.loc[round(float(d), 4)] = list(adf_res[:4]) + [adf_res[4]["5%"]] + [corr]

    # 保存结果
    csv_path = os.path.join(cfg.out_dir, f"{cfg.inst_name}_testMinFFD.csv")
    png_path = os.path.join(cfg.out_dir, f"{cfg.inst_name}_testMinFFD.png")
    out.to_csv(csv_path)

    if not out.empty:
        # 绘图：左轴 ADF 统计量，右轴相关系数
        ax = out[["adfStat", "corr"]].astype(float).plot(secondary_y="adfStat")
        plt.axhline(out["95% conf"].astype(float).mean(), linewidth=1, color="r", linestyle="dotted")
        plt.title(f"Min FFD d search ({cfg.inst_name})")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.show()


def main() -> None:
    """
    章节 5 的一站式示例：
    - 画出权重形状（5.1）
    - 对 BTCUSDT 做 FFD 分数阶微分（5.2/5.3）
    - 搜索通过 ADF 的最小 d 并画出 ADF/corr 曲线（5.4）
    """
    cfg5 = Ch5Config()

    # # 5.1：权重示意图（可选）
    # print("Plotting weight shapes for d in [0,1] and [1,2] ...")
    # plotWeights(dRange=[0, 1], nPlots=11, size=6)
    # plotWeights(dRange=[1, 2], nPlots=11, size=6)

    # 5.2/5.3/5.4：用与第 3 章相同的 BTCUSDT 数据做 FFD + ADF 搜索
    print("Loading close prices from Chapter 3 data ...")
    close = load_close_from_ch3(cfg5.data_cfg)
    print(f"Loaded close series length: {len(close)}")

    print("Running FFD + ADF grid search over d in [0,1] ...")
    plotMinFFD_from_series(close, cfg5)
    print(f"Results saved under: {cfg5.out_dir}")


if __name__ == "__main__":
    main()


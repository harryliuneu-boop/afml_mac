import argparse
import glob
import os
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BarType = Literal["time", "volume", "dollar"]


def load_trades_from_folder(folder: str, limit_files: int | None = None) -> pd.DataFrame:
    """
    读取某个目录下的 Binance trades/aggTrades 数据（csv 或 zip），合并成一个 DataFrame。

    要求数据至少包含以下列之一：
    - time 或 timestamp: 毫秒时间戳
    - price: 成交价格
    - qty 或 quantity: 成交数量（以标的资产计，例如 BTC）
    - quoteQty（可选）: 计价货币成交额（例如 USDT），如果存在则可直接作为美元成交额使用
    """
    pattern_csv = os.path.join(folder, "**", "*.csv")
    pattern_zip = os.path.join(folder, "**", "*.zip")
    files = sorted(glob.glob(pattern_csv, recursive=True) + glob.glob(pattern_zip, recursive=True))

    if not files:
        raise FileNotFoundError(f"在目录 {folder} 下没有找到 csv/zip 成交数据文件。")

    if limit_files is not None:
        files = files[:limit_files]

    dfs: list[pd.DataFrame] = []
    for f in files:
        # Binance 的历史 trades/aggTrades 文件往往是无表头的纯数据，需要手动指定列名
        df = pd.read_csv(f, header=None)
        print(f'{f}:{df.shape}')
        # 常见 7 列 trades 格式: tradeId, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
        if df.shape[1] == 7:
            df.columns = [
                "trade_id",
                "price",
                "qty",
                "quoteQty",
                "time",
                "isBuyerMaker",
                "isBestMatch",
            ]
        # 如果后续使用 aggTrades 或其他格式，可在这里扩展列名映射
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # 统一列名
    if "time" in data.columns:
        ts_col = "time"
    elif "timestamp" in data.columns:
        ts_col = "timestamp"
    else:
        raise KeyError("数据中找不到 time 或 timestamp 列。")

    if "qty" in data.columns:
        qty_col = "qty"
    elif "quantity" in data.columns:
        qty_col = "quantity"
    else:
        raise KeyError("数据中找不到 qty 或 quantity 列。")

    if "price" not in data.columns:
        raise KeyError("数据中找不到 price 列。")

    data = data.rename(
        columns={
            ts_col: "timestamp",
            qty_col: "qty",
        }
    )

    # 毫秒时间戳转为 datetime
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)

    # 计算美元成交额
    if "quoteQty" in data.columns:
        data["dollar"] = data["quoteQty"].astype(float)
    else:
        data["dollar"] = data["price"].astype(float) * data["qty"].astype(float)

    data["price"] = data["price"].astype(float)
    data["qty"] = data["qty"].astype(float)

    return data[["timestamp", "price", "qty", "dollar"]]


def build_time_bars(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    构建固定时间间隔的分时线（Time bars），使用收盘价计算收益。
    freq 例如 '1min', '5min', '15min' 等。
    """
    trades = trades.set_index("timestamp")
    ohlcv = trades["price"].resample(freq).ohlc()
    vol = trades["qty"].resample(freq).sum()
    dollar = trades["dollar"].resample(freq).sum()

    bars = ohlcv.copy()
    bars["volume"] = vol
    bars["dollar"] = dollar
    bars = bars.dropna(subset=["close"])
    return bars.reset_index()


def _build_value_bars(trades: pd.DataFrame, target_value: float, value_col: str) -> pd.DataFrame:
    """
    构建以某个累积指标为基准的 bars（成交量线 / 美元线）。
    value_col 为 'qty' 或 'dollar'。
    """
    if target_value <= 0:
        raise ValueError("target_value 必须为正数。")

    v = trades[value_col].to_numpy()
    cumsum_v = np.cumsum(v)
    bar_index = (cumsum_v // target_value).astype(int)

    df = trades.copy()
    df["bar_index"] = bar_index

    grouped = df.groupby("bar_index", as_index=False)
    bars = grouped.agg(
        timestamp=("timestamp", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        dollar=("dollar", "sum"),
        value_sum=(value_col, "sum"),
    )

    # 去掉由于不满一个 target_value 而产生的最后一根残缺 bar（可选）
    bars = bars[bars["value_sum"] >= target_value * 0.5].copy()
    bars = bars.drop(columns=["value_sum"])
    return bars


def build_volume_bars(trades: pd.DataFrame, vol_per_bar: float) -> pd.DataFrame:
    """
    构建成交量线（Volume bars），每根 bar 目标累计成交量约为 vol_per_bar。
    """
    return _build_value_bars(trades, target_value=vol_per_bar, value_col="qty")


def build_dollar_bars(trades: pd.DataFrame, dollar_per_bar: float) -> pd.DataFrame:
    """
    构建美元线（Dollar bars），每根 bar 目标累计美元成交额约为 dollar_per_bar。
    """
    return _build_value_bars(trades, target_value=dollar_per_bar, value_col="dollar")


def compute_log_returns(bars: pd.DataFrame) -> pd.Series:
    close = bars["close"].astype(float)
    r = np.log(close / close.shift(1))
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


def weekly_bar_count(bars: pd.DataFrame) -> pd.Series:
    """
    每周产生多少根 K 线。
    """
    dt = pd.to_datetime(bars["timestamp"])
    week = dt.dt.to_period("W").dt.start_time
    counts = pd.Series(1, index=week).groupby(level=0).sum()
    return counts


def serial_correlation(r: pd.Series, lag: int = 1) -> float:
    """
    计算收益率的序列相关性（自相关系数）。
    """
    return r.autocorr(lag=lag)


def monthly_variance(r: pd.Series, index_ts: pd.Series) -> pd.Series:
    """
    按月计算收益率方差。
    """
    ts = pd.to_datetime(index_ts.loc[r.index])
    month = ts.dt.to_period("M")
    var = pd.Series(r.values, index=month).groupby(level=0).var()
    return var


def jarque_bera_test(r: pd.Series) -> Tuple[float, float]:
    """
    雅克-贝拉正态性检验，返回 (JB 统计量, p-value)。
    """
    jb, p = stats.jarque_bera(r)
    return jb, p


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_weekly_counts(bars_dict: dict[str, pd.DataFrame], output_dir: str) -> None:
    ensure_dir(output_dir)
    counts_frames: list[pd.Series] = []
    for name, bars in bars_dict.items():
        c = weekly_bar_count(bars).rename(name)
        counts_frames.append(c)
    if not counts_frames:
        return
    counts_df = pd.concat(counts_frames, axis=1)

    plt.figure(figsize=(10, 6))
    for col in counts_df.columns:
        plt.plot(counts_df.index, counts_df[col], label=col)
    plt.xlabel("Week")
    plt.ylabel("Number of bars")
    plt.title("Weekly bar counts (time / volume / dollar)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "weekly_bar_counts.png"))
    plt.close()


def plot_monthly_variance(bars_dict: dict[str, pd.DataFrame], output_dir: str) -> None:
    ensure_dir(output_dir)
    var_frames: list[pd.Series] = []
    for name, bars in bars_dict.items():
        r = compute_log_returns(bars)
        var = monthly_variance(r, index_ts=bars["timestamp"]).rename(name)
        var_frames.append(var)
    if not var_frames:
        return
    var_df = pd.concat(var_frames, axis=1)

    plt.figure(figsize=(10, 6))
    for col in var_df.columns:
        plt.plot(var_df.index.to_timestamp(), var_df[col], marker="o", label=col)
    plt.xlabel("Month")
    plt.ylabel("Variance of returns")
    plt.title("Monthly return variance (time / volume / dollar)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "monthly_return_variance.png"))
    plt.close()


def plot_return_histograms(bars_dict: dict[str, pd.DataFrame], output_dir: str) -> None:
    ensure_dir(output_dir)
    for name, bars in bars_dict.items():
        r = compute_log_returns(bars)
        if r.empty:
            continue
        mu = r.mean()
        sigma = r.std(ddof=1)

        plt.figure(figsize=(8, 5))
        plt.hist(r, bins=100, density=True, alpha=0.6, label="Empirical")

        if sigma > 0:
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            y = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, y, "r-", lw=2, label="Normal fit")

        plt.title(f"Return distribution ({name} bars)")
        plt.xlabel("Log return")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"returns_hist_{name}.png"))
        plt.close()


def plot_price_bars(bars_dict: dict[str, pd.DataFrame], output_dir: str, max_bars: int = 2000) -> None:
    """
    为每种 bar 类型画一张简化版 K 线图（蜡烛图），展示价格随 bar 序列的变化。
    为避免图片过重，默认只画前 max_bars 根。
    """
    ensure_dir(output_dir)
    for name, bars in bars_dict.items():
        if bars.empty:
            continue
        df = bars.copy().reset_index(drop=True)
        if len(df) > max_bars:
            df = df.iloc[:max_bars]

        x = np.arange(len(df))

        plt.figure(figsize=(12, 6))
        for i, row in df.iterrows():
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            color = "red" if c >= o else "green"
            # 上下影线
            plt.vlines(x[i], l, h, color=color, linewidth=0.8)
            # 实体
            plt.vlines(x[i], min(o, c), max(o, c), color=color, linewidth=3.0)

        plt.title(f"Price bars ({name} bars)")
        plt.xlabel("Bar index (first ~{} bars)".format(len(df)))
        plt.ylabel("Price")
        plt.tight_layout()
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(output_dir, f"kline_{name}.png"))
        plt.close()


def run_full_analysis(
    trades_folder: str,
    time_freq: str = "1min",
    vol_per_bar: float = 10.0,
    dollar_per_bar: float = 10_000.0,
    output_dir: str = "plots",
) -> None:
    """
    对同一批成交数据，分别构建：
    - 时间线（time bars）
    - 成交量线（volume bars）
    - 美元线（dollar bars）

    并输出 / 绘图：
    1) 每周 K 线数量统计的稳定性
    2) 收益率的序列相关性
    3) 按月收益率方差
    4) Jarque-Bera 正态性检验
    同时生成对应图像：
    - 周 K 线数量对比折线图
    - 月度方差折线图
    - 三种 bars 收益率分布直方图 + 正态拟合曲线
    - 三种 bars 对应的价格 K 线图（简化蜡烛图）
    """
    trades = load_trades_from_folder(trades_folder)

    time_bars = build_time_bars(trades, freq=time_freq)
    vol_bars = build_volume_bars(trades, vol_per_bar=vol_per_bar)
    dollar_bars = build_dollar_bars(trades, dollar_per_bar=dollar_per_bar)

    bars_dict = {
        "time": time_bars,
        "volume": vol_bars,
        "dollar": dollar_bars,
    }

    print("=== 每周 K 线数量（越平稳越好） ===")
    for name, bars in bars_dict.items():
        counts = weekly_bar_count(bars)
        print(f"\n[{name} bars] 周 K 数量描述统计：")
        print(counts.describe())

    print("\n=== 收益率序列相关性 r_t vs r_{t-1}（越接近 0 越好） ===")
    for name, bars in bars_dict.items():
        r = compute_log_returns(bars)
        rho = serial_correlation(r, lag=1)
        print(f"[{name} bars] lag-1 自相关系数: {rho:.6f}  （样本数 {len(r)}）")

    print("\n=== 按月收益率方差（越平稳越好） ===")
    for name, bars in bars_dict.items():
        r = compute_log_returns(bars)
        monthly_var = monthly_variance(r, index_ts=bars["timestamp"])
        print(f"\n[{name} bars] 月度方差描述统计：")
        print(monthly_var.describe())

    print("\n=== Jarque-Bera 正态性检验（p-value 越大越接近正态） ===")
    for name, bars in bars_dict.items():
        r = compute_log_returns(bars)
        jb, p = jarque_bera_test(r)
        print(f"[{name} bars] JB 统计量 = {jb:.3f}, p-value = {p:.4f}, 样本数 = {len(r)}")

    # 画图并保存
    if output_dir:
        plot_weekly_counts(bars_dict, output_dir=output_dir)
        plot_monthly_variance(bars_dict, output_dir=output_dir)
        plot_return_histograms(bars_dict, output_dir=output_dir)
        plot_price_bars(bars_dict, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 Binance 成交数据构建时间线 / 成交量线 / 美元线，"
            "并进行周 K 数量、序列相关性、月度方差和 Jarque-Bera 检验分析。"
        )
    )
    parser.add_argument(
        "--trades-folder",
        type=str,
        required=True,
        help=(
            "本地成交数据所在目录，例如："
            " binance-public-data/data/spot/daily/trades/BTCUSDT "
            "或 aggTrades 对应目录。"
        ),
    )
    parser.add_argument(
        "--time-freq",
        type=str,
        default="1min",
        help="时间线 bar 的时间周期（pandas offset），默认 1min。",
    )
    parser.add_argument(
        "--vol-per-bar",
        type=float,
        default=10.0,
        help="成交量线中每根 bar 的目标累计成交量（单位：标的资产数量，如 BTC）。",
    )
    parser.add_argument(
        "--dollar-per-bar",
        type=float,
        default=10_000.0,
        help="美元线中每根 bar 的目标累计美元成交额（单位：USDT）。",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="仅使用前 N 个文件做测试，调试用，可不填。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="输出图像保存目录，默认 'plots'（相对当前工作目录）。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.limit_files is not None:
        # 为了少改函数签名，这里简单地设置环境变量并在 load 中使用 limit_files
        # 实际用的时候建议直接在 run_full_analysis 里增加参数传递。
        print("当前版本不直接使用 --limit-files，请在需要时手动截取数据目录中的文件。")
    run_full_analysis(
        trades_folder=args.trades_folder,
        time_freq=args.time_freq,
        vol_per_bar=args.vol_per_bar,
        dollar_per_bar=args.dollar_per_bar,
        output_dir=args.output_dir,
    )

# python crypto_bar_analysis.py --trades-folder btc_data/data/spot/daily/trades/BTCUSDT/2021-01-01_2021-01-10 --time-freq 1min --vol-per-bar 100 --dollar-per-bar 1e7 --output-dir plots
import argparse
import os

import pandas as pd

from crypto_bar_analysis import (
    build_dollar_bars,
    build_time_bars,
    build_volume_bars,
    load_trades_from_folder,
)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """
    写 parquet。优先用 pyarrow 引擎；如果环境缺少依赖，给出明确报错提示。
    """
    try:
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except ImportError as e:
        raise ImportError(
            "写 parquet 需要安装 pyarrow。请在你的环境中执行：pip install pyarrow"
        ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 Binance 成交数据，构建 time/volume/dollar bars 并保存为 parquet。"
    )
    parser.add_argument(
        "--trades-folder",
        type=str,
        required=True,
        help="本地成交数据目录（包含 csv/zip，可递归）。",
    )
    parser.add_argument(
        "--time-freq",
        type=str,
        default="15min",
        help="时间 bars 的周期（pandas offset），默认 15min。",
    )
    parser.add_argument(
        "--vol-per-bar",
        type=float,
        default=100.0,
        help="成交量 bars 每根累计 qty（单位：标的资产数量），默认 100。",
    )
    parser.add_argument(
        "--dollar-per-bar",
        type=float,
        default=1e7,
        help="美元 bars 每根累计 dollar（单位：USDT），默认 1e7。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bars_parquet",
        help="输出目录，默认 bars_parquet。",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="输出文件名前缀（可选），例如 BTCUSDT_2021Q1_。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    trades = load_trades_from_folder(args.trades_folder)

    time_bars = build_time_bars(trades, freq=args.time_freq)
    vol_bars = build_volume_bars(trades, vol_per_bar=args.vol_per_bar)
    dollar_bars = build_dollar_bars(trades, dollar_per_bar=args.dollar_per_bar)

    time_path = os.path.join(args.output_dir, f"{args.prefix}time_bars.parquet")
    vol_path = os.path.join(args.output_dir, f"{args.prefix}vol_bars.parquet")
    dollar_path = os.path.join(args.output_dir, f"{args.prefix}dollar_bars.parquet")

    write_parquet(time_bars, time_path)
    write_parquet(vol_bars, vol_path)
    write_parquet(dollar_bars, dollar_path)

    print("已写出 parquet：")
    print(f"- time_bars   -> {time_path}   (rows={len(time_bars)})")
    print(f"- vol_bars    -> {vol_path}    (rows={len(vol_bars)})")
    print(f"- dollar_bars -> {dollar_path} (rows={len(dollar_bars)})")


if __name__ == "__main__":
    main()

# /Users/liuhaoran/Documents/miniconda/miniconda/envs/btc/bin/python \
#   build_bars_to_parquet.py \
#   --trades-folder btc_data/data/spot/daily/trades/BTCUSDT/2021-01-01_2021-01-10 \
#   --time-freq 15min \
#   --vol-per-bar 100 \
#   --dollar-per-bar 1e7 \
#   --output-dir bars_parquet \
#   --prefix BTCUSDT_2021-01-01_2021-01-10_
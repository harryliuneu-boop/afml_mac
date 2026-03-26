import os
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


EventsKind = Literal["index", "timestamp", "pseudo_datetimeindex"]

"""
零参数版本（把参数写死在这里，直接运行脚本即可）

你只需要改下面这几个配置变量：
"""

# === 必填：dollar_bars parquet 路径 ===
DOLLAR_BARS_PARQUET = (
    "bars_parquet/BTCUSDT_2021-01-01_2021-01-10_dollar_bars.parquet"
)

# === events 输入（二选一）===
# 1) 从文件读取（.csv/.parquet/.feather），要求有 EVENTS_COL 这一列
EVENTS_PATH = "events.csv"  # 例如：Notebook 里保存出来的 events.csv
EVENTS_COL = "event"
# 2) 直接在这里写整数索引（优先级更高；不为空就会忽略 EVENTS_PATH）
EVENTS_INLINE = ""  # 例如 "10,50,75,112"

# === events 类型 ===
# - "index": 整数位置（0..len(bars)-1）
# - "timestamp": 真实时间戳（必须能和 dollar_bars 的 timestamp 精确匹配）
# - "pseudo_datetimeindex": 1970-...伪时间戳（你之前那种，底层 ns 数值就是整数位置）
EVENTS_KIND: EventsKind = "pseudo_datetimeindex"

# === dollar_bars 列名 ===
TS_COL = "timestamp"
PRICE_COL = "close"

# === 若没有提供 events，则自动计算（CUSUM）===
AUTO_EVENTS_ENABLED = True
# 用 price 的 pct_change() 算收益率波动率（跟你 notebook 一致）
VOL_WINDOW = 100
# h 的选择：mean=vol.mean(), last=vol.iloc[-1]
H_MODE: Literal["mean", "last"] = "mean"
# 阈值倍数（有时候需要放大/缩小）
H_MULTIPLIER = 1.0
# 使用 log return 还是简单 return
RETURN_MODE: Literal["log", "pct"] = "pct"

# === 画图样式（解决红点糊成一片）===
# events 的展示方式：
# - "scatter": 画点（默认）
# - "rug": 在图底部画短竖线（更清晰）
EVENT_PLOT_MODE: Literal["scatter", "rug"] = "rug"
# scatter 样式
EVENT_MARKER_SIZE = 6  # 点大小（s=面积），建议 3~10
EVENT_ALPHA = 0.25  # 透明度，越小越不挡线
# 是否对 events 做稀疏抽样（1=不抽样；2=每隔一个取一个；10=每10个取一个）
EVENT_STRIDE = 5

# 是否在静态图上标注部分 event 的编号（便于对齐到数据）
# 0 表示不标注；例如 50 表示每 50 个 event 标注一次“event_idx”
LABEL_EVERY_N_EVENT = 50

# === 输出 ===
OUTPUT_PNG = "plots/dollar_bars_events.png"
OUTPUT_SVG = "plots/dollar_bars_events.svg"  # 矢量图，放大不糊
OUTPUT_HTML = "plots/dollar_bars_events.html"  # 若安装了 plotly，会输出可交互图（推荐大数据用）
EVENTS_MAPPING_CSV = "plots/events_mapping.csv"  # event_idx -> timestamp/price 映射表
TITLE = "Dollar bars with events"


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"不支持的 events 文件格式: {path}（支持 .parquet/.csv/.feather）")


def parse_events(
    *,
    events_path: str | None,
    events_inline: str | None,
    kind: EventsKind,
    events_col: str,
) -> pd.Index:
    """
    返回一个 pd.Index，内容是：
    - kind=index: 整数位置（Int64Index）
    - kind=timestamp: 真实时间戳（DatetimeIndex）
    - kind=pseudo_datetimeindex: 1970-...伪时间戳（DatetimeIndex），其底层 ns 值就是整数位置
    """
    if events_inline:
        # 逗号分隔整数
        idx = [int(x.strip()) for x in events_inline.split(",") if x.strip()]
        return pd.Index(idx, dtype="int64")

    if not events_path:
        raise ValueError("必须提供 --events-path 或 --events 之一。")

    df = read_table(events_path)
    if events_col not in df.columns:
        raise KeyError(f"events 文件缺少列 {events_col!r}。可用列: {list(df.columns)}")

    s = df[events_col]

    if kind == "index":
        return pd.Index(pd.to_numeric(s, errors="raise").astype("int64"))

    if kind == "timestamp":
        return pd.DatetimeIndex(pd.to_datetime(s, errors="raise"))

    if kind == "pseudo_datetimeindex":
        # 你目前 notebook 里 events 类似：
        # DatetimeIndex(['1970-01-01 00:00:00.000000010', ...])
        # 这种其实是把整数 index 当作 ns 解释后的结果，底层 int64 ns 就是原 index。
        dt = pd.DatetimeIndex(pd.to_datetime(s, errors="raise"))
        # dt.asi8 -> int64 纳秒
        return pd.Index(dt.asi8.astype("int64"))

    raise ValueError(f"未知 kind: {kind}")


def cusum_events(price: pd.Series, h: float, return_mode: Literal["log", "pct"] = "pct") -> pd.Index:
    """
    简单 CUSUM 事件过滤器：
    - price: 价格序列（按时间排序）
    - h: 阈值（越大事件越少）
    返回：触发事件的位置索引（整数位置）
    """
    p = pd.to_numeric(price, errors="raise").astype(float).reset_index(drop=True)
    if return_mode == "log":
        r = np.log(p).diff()
    else:
        r = p.pct_change()

    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    s_pos = 0.0
    s_neg = 0.0
    events: list[int] = []
    for i in range(1, len(r)):
        x = float(r[i])
        s_pos = max(0.0, s_pos + x)
        s_neg = min(0.0, s_neg + x)
        if s_pos > h:
            s_pos = 0.0
            events.append(i)
        elif s_neg < -h:
            s_neg = 0.0
            events.append(i)
    return pd.Index(events, dtype="int64")


def main() -> None:
    dollar_bars = pd.read_parquet(DOLLAR_BARS_PARQUET)
    if TS_COL not in dollar_bars.columns:
        raise KeyError(f"dollar_bars 缺少时间列 {TS_COL!r}")
    if PRICE_COL not in dollar_bars.columns:
        raise KeyError(f"dollar_bars 缺少价格列 {PRICE_COL!r}")

    ts = pd.to_datetime(dollar_bars[TS_COL])
    price = pd.to_numeric(dollar_bars[PRICE_COL], errors="raise")

    events_inline = EVENTS_INLINE if EVENTS_INLINE else None
    events_path = EVENTS_PATH if (EVENTS_PATH and os.path.exists(EVENTS_PATH)) else None

    if events_inline or events_path:
        events_index = parse_events(
            events_path=events_path,
            events_inline=events_inline,
            kind=EVENTS_KIND,
            events_col=EVENTS_COL,
        )
    else:
        if not AUTO_EVENTS_ENABLED:
            raise FileNotFoundError(
                f"找不到 events 文件 {EVENTS_PATH!r}，并且 EVENTS_INLINE 为空，同时 AUTO_EVENTS_ENABLED=False。"
            )
        vol = price.pct_change().rolling(window=VOL_WINDOW).std()
        if H_MODE == "last":
            h = float(vol.dropna().iloc[-1])
        else:
            h = float(vol.mean())
        h = h * float(H_MULTIPLIER)
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f"自动计算得到的 h 无效：{h}（请检查数据或调整 VOL_WINDOW/H_MODE）")
        events_index = cusum_events(price, h=h, return_mode=RETURN_MODE)
        print(f"[AUTO] vol_window={VOL_WINDOW} H_MODE={H_MODE} h={h:.8f} events={len(events_index)}")

    out_dir = os.path.dirname(OUTPUT_PNG) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 统一成整数索引 event_idx_full（便于导出 mapping / 画交互图）
    if EVENTS_KIND == "timestamp":
        event_times = pd.DatetimeIndex(events_index)
        idx_map = pd.Index(ts)
        mask = idx_map.isin(event_times)
        event_idx_full = pd.Index(np.flatnonzero(mask), dtype="int64")
        print(f"events(时间) 总数={len(event_times)}，成功对齐到 bars 的数量={len(event_idx_full)}")
    else:
        event_idx_full = pd.Index(events_index).astype("int64")
        event_idx_full = event_idx_full[(event_idx_full >= 0) & (event_idx_full < len(dollar_bars))]
        print(f"events(索引) 标注数量={len(event_idx_full)} / bars={len(dollar_bars)}（未抽样）")

    # 导出 mapping：方便你精确对齐 event 在哪根 bar
    mapping = pd.DataFrame(
        {
            "event_idx": event_idx_full.to_numpy(),
            "timestamp": ts.iloc[event_idx_full].to_numpy(),
            "price": price.iloc[event_idx_full].to_numpy(),
        }
    )
    mapping.to_csv(EVENTS_MAPPING_CSV, index=False)
    print(f"已写出 events mapping: {EVENTS_MAPPING_CSV} (rows={len(mapping)})")

    # 静态图：SVG 放大不糊，PNG 只是预览
    plt.figure(figsize=(16, 7))
    plt.plot(ts, price, label="dollar bars close", alpha=0.9, linewidth=0.9)

    event_idx_plot = event_idx_full
    if EVENT_STRIDE > 1:
        event_idx_plot = event_idx_plot[::EVENT_STRIDE]

    if EVENT_PLOT_MODE == "scatter":
        plt.scatter(
            ts.iloc[event_idx_plot],
            price.iloc[event_idx_plot],
            color="red",
            s=EVENT_MARKER_SIZE,
            alpha=EVENT_ALPHA,
            label="events",
        )
    else:
        ymin, ymax = plt.ylim()
        y0 = ymin + 0.02 * (ymax - ymin)
        y1 = ymin + 0.09 * (ymax - ymin)
        plt.vlines(
            ts.iloc[event_idx_plot],
            y0,
            y1,
            color="red",
            alpha=min(0.7, max(0.05, EVENT_ALPHA * 2)),
            linewidth=0.7,
            label="events",
        )

    if LABEL_EVERY_N_EVENT and LABEL_EVERY_N_EVENT > 0:
        for k in range(0, len(event_idx_full), LABEL_EVERY_N_EVENT):
            i = int(event_idx_full[k])
            plt.annotate(
                str(i),
                (ts.iloc[i], float(price.iloc[i])),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color="darkred",
                alpha=0.9,
            )

    print(f"events(绘图) 数量={len(event_idx_plot)} / 总events={len(event_idx_full)}（stride={EVENT_STRIDE}）")

    plt.title(TITLE)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220)
    plt.savefig(OUTPUT_SVG)
    plt.close()

    print(f"已保存图像: {OUTPUT_PNG}")
    print(f"已保存矢量图: {OUTPUT_SVG}")

    # 可交互图（大数据量推荐）
    try:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=price, mode="lines", name="close", line=dict(width=1)))

        hover_text = [
            f"event_idx={int(i)}<br>time={ts.iloc[int(i)]}<br>price={float(price.iloc[int(i)]):.2f}"
            for i in event_idx_full.to_list()
        ]
        fig.add_trace(
            go.Scatter(
                x=ts.iloc[event_idx_full],
                y=price.iloc[event_idx_full],
                mode="markers",
                name="events",
                marker=dict(size=4, color="red", opacity=0.35),
                text=hover_text,
                hoverinfo="text",
            )
        )
        fig.update_layout(
            title=TITLE,
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
            height=650,
        )
        fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
        print(f"已保存可交互图: {OUTPUT_HTML}")
    except Exception as e:
        print(f"未生成可交互图（可选）：{e}")


if __name__ == "__main__":
    main()


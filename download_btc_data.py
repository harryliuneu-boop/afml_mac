"""
Binance data.binance.vision 按日下载 trades/aggTrades（daily zip）。

特性：
- 近 N 年按天下载（默认 5 年）
- 并行下载（ThreadPoolExecutor）
- 已存在且非空 zip → 跳过；可 --only-missing 只提交缺失日任务
- 失败重试：单次下载内最多 retries 次，指数退避
- 下载先写入 .zip.tmp，成功后 replace；失败删 .tmp，避免残缺 zip
- 运行前/后扫描本地：缺失日、空文件、残留 .tmp，并可选清理
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import requests

BASE_URL = "https://data.binance.vision/data"

# ---------- 默认配置（也可命令行覆盖）----------
DEFAULT_STORE = "/Users/liuhaoran/Documents/program/afml/download"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_MARKET = "spot"
DEFAULT_DATA_KIND = "trades"
DEFAULT_YEARS = 5
DEFAULT_WORKERS = 8
DEFAULT_RETRIES = 5


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_remote_rel_path(
    market_type: str, freq: str, data_kind: str, symbol: str, suffix: str
) -> str:
    if market_type == "spot":
        base = f"spot/{freq}/{data_kind}/{symbol}"
    else:
        base = f"futures/{market_type}/{freq}/{data_kind}/{symbol}"
    filename = f"{symbol}-{data_kind}-{suffix}.zip"
    return f"{base}/{filename}"


def day_to_zip_paths(
    store_root: Path,
    market_type: str,
    data_kind: str,
    symbol: str,
    d: date,
) -> tuple[Path, Path]:
    """返回 (最终 zip 路径, 临时 .zip.tmp 路径)。"""
    day_suffix = d.strftime("%Y-%m-%d")
    rel = build_remote_rel_path(market_type, "daily", data_kind, symbol, day_suffix)
    out = store_root / rel
    tmp = out.with_suffix(out.suffix + ".tmp")
    return out, tmp


def _remove_if_exists(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except OSError:
        pass


@dataclass
class CoverageReport:
    """预期日期区间内，本地文件与「应有数据」的对比。"""

    expected_days: int
    complete: list[date]
    missing: list[date]
    empty_zip: list[date]
    stale_tmp: list[tuple[date, Path]]

    @property
    def complete_count(self) -> int:
        return len(self.complete)

    @property
    def missing_count(self) -> int:
        return len(self.missing)


def scan_coverage(
    days: list[date],
    store_root: Path,
    market_type: str,
    data_kind: str,
    symbol: str,
    *,
    cleanup_stale_tmp: bool = False,
    cleanup_empty_zip: bool = False,
) -> CoverageReport:
    """
    扫描 [days] 内每天应对应的 zip 是否存在且非空。
    - 非空 zip → 视为已完整
    - 仅存在 .tmp 或 0 字节 zip → 视为缺失（可下载补齐）
    """
    complete: list[date] = []
    missing: list[date] = []
    empty_zip: list[date] = []
    stale_tmp: list[tuple[date, Path]] = []

    for d in days:
        out, tmp = day_to_zip_paths(store_root, market_type, data_kind, symbol, d)

        has_zip = out.exists() and out.stat().st_size > 0
        has_tmp = tmp.exists()
        zip_zero = out.exists() and out.stat().st_size == 0

        if has_zip:
            complete.append(d)
            if has_tmp:
                stale_tmp.append((d, tmp))
                if cleanup_stale_tmp:
                    _remove_if_exists(tmp)
            continue

        if zip_zero:
            empty_zip.append(d)
            missing.append(d)
            if cleanup_empty_zip:
                _remove_if_exists(out)
            if has_tmp and cleanup_stale_tmp:
                _remove_if_exists(tmp)
            continue

        if has_tmp:
            stale_tmp.append((d, tmp))
            if cleanup_stale_tmp:
                _remove_if_exists(tmp)

        missing.append(d)

    return CoverageReport(
        expected_days=len(days),
        complete=sorted(complete),
        missing=sorted(missing),
        empty_zip=sorted(empty_zip),
        stale_tmp=stale_tmp,
    )


def print_coverage_report(title: str, rep: CoverageReport, *, max_list: int = 15) -> None:
    print(f"\n=== {title} ===")
    print(f"预期天数: {rep.expected_days}")
    print(f"本地完整(非空 zip): {rep.complete_count}")
    print(f"仍缺失(需下载或 404): {rep.missing_count}")
    if rep.empty_zip:
        print(f"  其中 0 字节 zip: {len(rep.empty_zip)} 天")
    if rep.stale_tmp:
        print(f"  残留 .zip.tmp: {len(rep.stale_tmp)} 个（建议清理后重试）")
    if rep.missing and rep.missing_count <= max_list:
        print(f"缺失日期: {[str(x) for x in rep.missing]}")
    elif rep.missing:
        head = [str(x) for x in rep.missing[:max_list]]
        print(f"缺失日期(前 {max_list} 个): {head} ... 共 {rep.missing_count} 天")


def download_one_daily_zip(
    url: str,
    out_path: Path,
    *,
    session: requests.Session | None = None,
    retries: int = DEFAULT_RETRIES,
    chunk_size: int = 1024 * 1024,
) -> tuple[str, str]:
    """
    下载单日 zip（含失败重试）。

    重试逻辑：同一 URL 最多尝试 retries 次；每次失败后删除 .tmp、指数退避再请求。
    返回 (status, message)：
    - skip: 已存在非空 zip
    - ok: 下载成功
    - miss: 404（远端无该日文件，不重试）
    - fail: 重试耗尽仍失败（.tmp 已清理）
    """
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip", str(out_path)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    _remove_if_exists(tmp_path)

    sess = session or requests.Session()
    err: Exception | None = None

    for attempt in range(retries):
        try:
            with sess.get(url, stream=True, timeout=60) as resp:
                if resp.status_code == 404:
                    _remove_if_exists(tmp_path)
                    return "miss", url

                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                _remove_if_exists(tmp_path)
                raise RuntimeError("empty download")

            tmp_path.replace(out_path)
            return "ok", str(out_path)

        except Exception as e:
            err = e
            _remove_if_exists(tmp_path)
            time.sleep(min(2**attempt, 16))

    return "fail", f"{url} -> {err}"


def collect_date_range(years: int, end_exclusive: date | None = None) -> tuple[date, date]:
    if end_exclusive is None:
        end = date.today() - timedelta(days=1)
    else:
        end = end_exclusive
    start = end - timedelta(days=int(365.25 * years))
    return start, end


def parse_args() -> argparse.Namespace | None:
    if len(sys.argv) > 1:
        p = argparse.ArgumentParser(description="并行按日下载 Binance daily trades/aggTrades zip。")
        p.add_argument("-t", "--market-type", choices=["spot", "um", "cm"], default=DEFAULT_MARKET)
        p.add_argument("-s", "--symbol", default=DEFAULT_SYMBOL)
        p.add_argument(
            "-d",
            "--data-kind",
            choices=["trades", "aggTrades"],
            default=DEFAULT_DATA_KIND,
        )
        p.add_argument("--years", type=float, default=DEFAULT_YEARS, help="结束日往前推多少年，默认 5")
        p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行线程数，默认 8")
        p.add_argument(
            "--retries",
            type=int,
            default=DEFAULT_RETRIES,
            help="单日下载失败时的最大重试次数（指数退避），默认 5",
        )
        p.add_argument(
            "--store-directory",
            default=os.environ.get("STORE_DIRECTORY", DEFAULT_STORE),
        )
        p.add_argument("--end-date", default="", help="结束日期 YYYY-MM-DD，默认昨天")
        p.add_argument(
            "--only-missing",
            action="store_true",
            help="只对「本地缺失」的日期发起下载（多次运行补全时更快）",
        )
        p.add_argument(
            "--check-only",
            action="store_true",
            help="只扫描本地缺失情况，不下载",
        )
        p.add_argument(
            "--cleanup-tmp",
            action="store_true",
            # default=True,
            help="扫描时删除：已有完整 zip 旁的残留 .tmp、以及缺失日的孤立 .tmp",
        )
        p.add_argument(
            "--cleanup-empty-zip",
            action="store_true",
            help="扫描时删除 0 字节 zip，便于下次重新下载",
        )
        return p.parse_args()
    return None


def main() -> None:
    cli = parse_args()

    if cli is not None:
        market_type = cli.market_type
        symbol = cli.symbol.upper()
        data_kind = cli.data_kind
        years = float(cli.years)
        workers = max(1, int(cli.workers))
        retries = max(1, int(cli.retries))
        store_root = Path(cli.store_directory).expanduser().resolve()
        only_missing = cli.only_missing
        check_only = cli.check_only
        cleanup_tmp = cli.cleanup_tmp
        cleanup_empty = cli.cleanup_empty_zip
        if cli.end_date:
            end = date.fromisoformat(cli.end_date)
        else:
            end = date.today() - timedelta(days=1)
        start = end - timedelta(days=int(365.25 * years))
    else:
        market_type = DEFAULT_MARKET
        symbol = DEFAULT_SYMBOL
        data_kind = DEFAULT_DATA_KIND
        workers = DEFAULT_WORKERS
        retries = DEFAULT_RETRIES
        store_root = Path(DEFAULT_STORE).expanduser().resolve()
        only_missing = True  # 无参数时多次运行优先只补缺失
        check_only = False
        cleanup_tmp = False
        cleanup_empty = False
        end = date.today() - timedelta(days=1)
        start, end = collect_date_range(DEFAULT_YEARS, end_exclusive=end)

    days = list(daterange(start, end))

    print(f"=== Binance daily {data_kind} ===")
    print(f"symbol={symbol} market={market_type}")
    print(f"日期范围: {start} ~ {end} （共 {len(days)} 天）")
    print(f"保存目录: {store_root}")
    print(f"并行 workers={workers} | 单日重试 retries={retries}")
    print(f"仅补缺失: {only_missing} | 只检查: {check_only}")

    rep_before = scan_coverage(
        days,
        store_root,
        market_type,
        data_kind,
        symbol,
        cleanup_stale_tmp=cleanup_tmp,
        cleanup_empty_zip=cleanup_empty,
    )
    print_coverage_report("运行前 · 本地数据检查（相对预期日期区间）", rep_before)

    if check_only:
        print("\n(--check-only) 已结束，未发起下载。")
        return

    if only_missing:
        days_to_run = rep_before.missing
        if not days_to_run:
            print("\n无缺失日，无需下载。")
            rep_after = scan_coverage(
                days,
                store_root,
                market_type,
                data_kind,
                symbol,
            )
            print_coverage_report("运行后 · 本地数据检查", rep_after)
            return
        print(f"\n本次将尝试下载 {len(days_to_run)} 个缺失日（已跳过 {rep_before.complete_count} 个完整日）。")
    else:
        days_to_run = days

    stats = {"skip": 0, "ok": 0, "miss": 0, "fail": 0}

    def task(d: date) -> tuple[date, str, str]:
        day_suffix = d.strftime("%Y-%m-%d")
        rel = build_remote_rel_path(market_type, "daily", data_kind, symbol, day_suffix)
        url = f"{BASE_URL}/{rel}"
        out = store_root / rel
        status, msg = download_one_daily_zip(url, out, session=requests.Session(), retries=retries)
        return d, status, msg

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(task, d): d for d in days_to_run}
        done_n = 0
        total = len(days_to_run)
        for fut in as_completed(futures):
            d = futures[fut]
            done_n += 1
            try:
                _, status, msg = fut.result()
            except Exception as e:
                stats["fail"] += 1
                print(f"[fail] {d} exception: {e}")
                continue

            stats[status] = stats.get(status, 0) + 1
            if status == "ok":
                print(f"[{done_n}/{total}] [ok] {d}")
            elif status == "skip":
                print(f"[{done_n}/{total}] [skip] {d}")
            elif status == "miss":
                print(f"[{done_n}/{total}] [miss] {d} (404)")
            else:
                print(f"[{done_n}/{total}] [fail] {d} {msg}")

    print("\n=== 本次任务统计 ===")
    print(f"skip(已存在): {stats['skip']}")
    print(f"ok(新下载):   {stats['ok']}")
    print(f"miss(无文件): {stats['miss']}")
    print(f"fail(失败):   {stats['fail']}")

    rep_after = scan_coverage(days, store_root, market_type, data_kind, symbol)
    print_coverage_report("运行后 · 本地数据检查（仍缺失的会列出）", rep_after)

    miss_after = rep_after.missing_count
    if miss_after > 0:
        miss_path = store_root / f"_missing_dates_{data_kind}_{symbol}.txt"
        try:
            miss_path.write_text("\n".join(str(x) for x in rep_after.missing) + "\n", encoding="utf-8")
            print(f"\n仍缺失日期已写入: {miss_path}")
        except OSError as e:
            print(f"\n写入缺失列表失败: {e}")

    print(f"\n目录: {store_root}")


if __name__ == "__main__":
    main()

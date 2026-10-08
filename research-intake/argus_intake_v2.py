#!/usr/bin/env python3
"""Argus public-data intake V2: data, NOT strategy or trading authority.

Inputs: read-only exchange directory queue and prior checkpoint from intake-data branch.
Outputs: bounded, reproducible 60d/5m OHLCV research batches as Actions artifacts
plus metadata/checkpoint candidates. NO Google Drive, Colab, broker, or Supabase writes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import pandas as pd

SCHEMA = "ARGUS_GITHUB_INTAKE_RESEARCH_BATCH_V2"
STATE_SCHEMA = "ARGUS_GITHUB_INTAKE_SHARD_STATE_V2"
ET = ZoneInfo("America/New_York")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_file(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finite(value, places=6):
    try:
        f = float(value)
        return round(f, places) if math.isfinite(f) else None
    except (TypeError, ValueError, OverflowError):
        return None


def clean_bars(raw: pd.DataFrame, cutoff_date):
    """Discard current day, non-RTH, invalid, and duplicate bars; never impute."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(), "NO_BARS"
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        return pd.DataFrame(), "INVALID_TIMESTAMP"
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
        return pd.DataFrame(), "INVALID_OHLCV_SCHEMA"
    frame = raw.tz_convert(ET).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")].copy()
    minute = frame.index.hour * 60 + frame.index.minute
    frame = frame[(minute >= 570) & (minute < 960) & (frame.index.date < cutoff_date)].copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    frame = frame[(frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    frame = frame[frame.Volume >= 0]
    frame = frame[(frame.High >= frame[["Open", "Close", "Low"]].max(axis=1)) &
                  (frame.Low <= frame[["Open", "Close", "High"]].min(axis=1))]
    return frame, ("HAS_RTH_DATA" if not frame.empty else "NO_VALID_RTH_BARS")


def summarize(sym, raw, cutoff_date):
    """Descriptive features ONLY, no predictions, V11 claims, or lookahead labels."""
    frame, status = clean_bars(raw, cutoff_date)
    meta = {
        "symbol": sym, "status": status, "rth_bars": int(len(frame)),
        "sessions_observed": 0, "complete_sessions": 0, "last_session_et": None,
        "last_close": None, "avg_dollar_volume_20_sessions": None,
        "mean_session_return_pct": None, "std_session_return_pct": None,
        "mean_absolute_5m_return_pct": None, "max_abs_5m_return_pct": None,
        "last_20_session_return_pct": None,
        "research_score": None,
        "use_for_trading": False,
    }
    bar_rows, daily_rows = [], []
    if frame.empty:
        return meta, bar_rows, daily_rows
    for ts, row in frame.iterrows():
        bar_rows.append({
            "symbol": sym, "utc": ts.tz_convert(timezone.utc).isoformat(),
            "et": ts.isoformat(), "o": finite(row.Open), "h": finite(row.High),
            "l": finite(row.Low), "c": finite(row.Close), "v": int(row.Volume)
        })
    for day, group in frame.groupby(frame.index.date, sort=True):
        observed = len(group)
        o = float(group.iloc[0].Open)
        c = float(group.iloc[-1].Close)
        dollar = (group.Close * group.Volume).sum()
        daily_rows.append({
            "symbol": sym, "session_et": str(day), "bars": observed,
            "is_near_complete_rth": bool(observed >= 60),
            "open": finite(o), "close": finite(c),
            "session_return_pct": finite(100 * (c / o - 1)) if o else None,
            "dollar_volume": finite(dollar, 2),
        })
    meta["sessions_observed"] = len(daily_rows)
    meta["complete_sessions"] = sum(1 for r in daily_rows if r["is_near_complete_rth"])
    meta["last_session_et"] = daily_rows[-1]["session_et"]
    meta["last_close"] = daily_rows[-1]["close"]
    full = [r for r in daily_rows if r["is_near_complete_rth"]]
    if len(full) >= 20:
        window = full[-20:]
        meta["avg_dollar_volume_20_sessions"] = finite(sum(r["dollar_volume"] for r in window) / 20, 2)
        meta["last_20_session_return_pct"] = finite(100 * (window[-1]["close"] / window[0]["close"] - 1))
        returns = [r["session_return_pct"] for r in window if r["session_return_pct"] is not None]
        meta["mean_session_return_pct"] = finite(sum(returns) / len(returns)) if returns else None
        if len(returns) > 1:
            mean = sum(returns) / len(returns)
            meta["std_session_return_pct"] = finite((sum((v-mean)**2 for v in returns)/(len(returns)-1))**0.5)
        price = window[-1]["close"]
        adv = meta["avg_dollar_volume_20_sessions"]
        age = (cutoff_date - datetime.fromisoformat(window[-1]["session_et"]).date()).days
        meta["last_complete_session_age_days"] = age
        if age > 7:
            meta["status"] = "STALE_HISTORY"
        elif price < 1:
            meta["status"] = "LOW_PRICE"
        elif adv is None or adv < 1_000_000:
            meta["status"] = "LOW_DOLLAR_VOLUME"
        else:
            meta["status"] = "RESEARCH_DATA_READY"
    else:
        meta["status"] = "INSUFFICIENT_COMPLETE_SESSIONS"
    # Intraday statistics computed *within* each session, never bridging overnight.
    # This is descriptive, NOT an after-the-fact prediction or V11 feature.
    intraday = frame.Close.groupby(frame.index.date).pct_change().dropna() * 100
    if len(intraday):
        meta["mean_absolute_5m_return_pct"] = finite(intraday.abs().mean())
        meta["max_abs_5m_return_pct"] = finite(intraday.abs().max())
    return meta, bar_rows, daily_rows


def get_history(symbol):
    import yfinance as yf
    return yf.Ticker(symbol).history(
        period="60d", interval="5m", auto_adjust=False, actions=False,
        prepost=False, timeout=30
    )


def source_canary():
    try:
        import yfinance as yf
        x = yf.Ticker("SPY").history(
            period="5d", interval="5m", auto_adjust=False,
            actions=False, prepost=False, timeout=25
        )
        return isinstance(x, pd.DataFrame) and len(x) >= 20 and {"Close", "Volume"}.issubset(x.columns)
    except Exception:
        return False


def run(args):
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    q = json.loads(Path(args.queue).read_text())
    intake = json.loads(Path(args.intake_status).read_text())
    assert q["schema"] == "ARGUS_GITHUB_RESEARCH_QUEUE_SHARD_V1"
    assert intake["schema"] == "ARGUS_GITHUB_RESEARCH_INTAKE_STATUS_V1"
    assert intake["state"] == "CATALOG_READY"
    assert q["source_digest_sha256"] == intake["source_digest_sha256"], "Queue digest mismatch"
    assert q["shard"] == args.shard and q["shards_total"] == 8, "Wrong queue/shard"
    symbols = q["symbols"]
    assert isinstance(symbols, list) and len(symbols) > 0
    assert 1 <= args.count <= 256 and 5 <= args.budget_minutes <= 330
    prior = {}
    if args.prior and Path(args.prior).exists():
        prior = json.loads(Path(args.prior).read_text())
        if prior.get("schema") != STATE_SCHEMA or prior.get("source_digest_sha256") != q["source_digest_sha256"]:
            prior = {}
    cursor = int(prior.get("cursor", 0)) % len(symbols)
    retry = [s for s in prior.get("retry_symbols", []) if s in symbols]
    retry = list(dict.fromkeys(retry))[:200]
    retry_take = retry[:min(max(1, args.count // 4), 8)]
    new_take_count = args.count - len(retry_take)
    new_syms = [symbols[(cursor + i) % len(symbols)] for i in range(new_take_count)]
    selected = list(dict.fromkeys(retry_take + new_syms))
    selected = selected[:args.count]
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + args.budget_minutes * 60 - 35
    cutoff_date = datetime.now(ET).date()  # Today is excluded, always.
    features = []
    attempted_new = 0
    successful_new = 0
    bars_seen = 0
    source_errors = 0
    output_features = path / "features.jsonl.gz"
    output_bars = path / "bars.jsonl.gz"
    output_sessions = path / "sessions.jsonl.gz"
    errors = {}
    with gzip.open(output_features, "wt", encoding="utf-8", compresslevel=6) as feat_out, \
         gzip.open(output_bars, "wt", encoding="utf-8", compresslevel=6) as bars_out, \
         gzip.open(output_sessions, "wt", encoding="utf-8", compresslevel=6) as daily_out:
        for sym in selected:
            if time.monotonic() >= deadline:
                break
            is_retry = sym in retry_take
            if not is_retry:
                attempted_new += 1
            try:
                raw = get_history(sym)
                feature, bar_rows, day_rows = summarize(sym, raw, cutoff_date)
            except Exception as exc:
                feature = {"symbol": sym, "status": "SOURCE_ERROR", "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                           "rth_bars": 0, "research_score": None, "use_for_trading": False}
                bar_rows, day_rows = [], []
                source_errors += 1
            features.append(feature)
            if feature["status"] != "SOURCE_ERROR" and not is_retry:
                successful_new += 1
            if feature["status"] == "SOURCE_ERROR":
                errors[sym] = feature.get("error", "unknown source error")
            bars_seen += len(bar_rows)
            feat_out.write(canonical(feature) + "\n")
            for item in bar_rows:
                bars_out.write(canonical(item) + "\n")
            for item in day_rows:
                daily_out.write(canonical(item) + "\n")
            print(sym, feature["status"], "5m_bars", len(bar_rows), flush=True)
            time.sleep(.65)
    # When a run produced zero bars, verify the source independently; a whole batch
    # of inaccessible tickers is not equivalent to a market-data-provider outage.
    canary_ok = source_canary() if bars_seen == 0 or source_errors > max(2, len(features)//2) else None
    source_ok = bars_seen > 0 or canary_ok is True
    # Do not advance the main queue if data source health is unverified.
    # A partially useful batch is retained as diagnostics; no state checkpoint.
    next_retry = [s for s in retry if s not in {f["symbol"] for f in features if f["status"] != "SOURCE_ERROR"}]
    next_retry += [f["symbol"] for f in features if f["status"] == "SOURCE_ERROR" and f["symbol"] not in next_retry]
    next_retry = next_retry[:200]
    counts = dict(Counter(f["status"] for f in features))
    # Cursor counts visited first-time symbols. Failed symbols are separately queued for retries.
    if source_ok:
        cursor_next = (cursor + attempted_new) % len(symbols)
    else:
        cursor_next = cursor
    run_id = str(args.run_id)
    manifest = {
        "schema": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(), "run_id": run_id,
        "shard": args.shard, "input_digest_sha256": q["source_digest_sha256"],
        "queue_size": len(symbols), "cursor_before": cursor, "cursor_after": cursor_next,
        "selected_symbols": [f["symbol"] for f in features], "counts": counts,
        "symbols_processed": len(features), "rth_five_minute_bars": bars_seen,
        "new_symbols_visited": attempted_new, "new_symbols_without_source_exception": successful_new,
        "source_errors": source_errors, "provider_canary_ok": canary_ok,
        "source_health_verified": bool(source_ok), "retry_symbols_pending": len(next_retry),
        "storage": {"features": "features.jsonl.gz", "bars": "bars.jsonl.gz", "sessions": "sessions.jsonl.gz"},
        "data_window": "Most recent 60 calendar days of source 5-minute bars, excluding current ET session",
        "source": "Yahoo Finance via yfinance, auto_adjust=False, actions=False; research use only",
        "authority": "RESEARCH_DATA_ONLY_NO_TRADING_AUTHORITY",
        "v11_validated": False, "model_scored": False, "drive_saved": False,
        "notices": ["Preprocessing is descriptive, not V11 parity or a trade recommendation.",
                    "Source revisions, corporate actions, sparse bars and timestamp quality require audit.",
                    "Files are retained in GitHub Actions artifacts for 30 days, not indefinitely."]
    }
    state = {
        "schema": STATE_SCHEMA, "updated_at_utc": manifest["generated_at_utc"],
        "source_digest_sha256": q["source_digest_sha256"], "shard": args.shard,
        "cursor": cursor_next, "queue_size": len(symbols),
        "retry_symbols": next_retry, "last_run_id": run_id,
        "last_counts": counts, "last_symbols_processed": len(features)
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (path / "next-state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    checksums = {name: digest_file(path / name) for name in
                 ("features.jsonl.gz", "bars.jsonl.gz", "sessions.jsonl.gz", "manifest.json", "next-state.json")}
    (path / "sha256.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    print("ARGUS_INTAKE_RESULT", canonical({"shard": args.shard, "run_id": run_id,
          "count": len(features), "bars": bars_seen, "source_ok": source_ok, "cursor": cursor_next}), flush=True)
    if not features or not source_ok:
        print("No research checkpoint: no completed results or market data source unverified.", flush=True)
        return 2
    return 0


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--intake-status", required=True)
    ap.add_argument("--prior", required=False)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--budget-minutes", type=int, default=45)
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local-test"))
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(cli()))

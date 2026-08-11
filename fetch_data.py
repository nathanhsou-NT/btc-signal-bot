"""
fetch_data.py
-------------
Baixa candles de 1 MINUTO de BTC/USDT da API publica da Binance (sem API key).

Por que 1 minuto e nao 15 minutos direto?
  Porque a decisao precisa sair no minuto 13 da vela M15 atual (2 min antes
  da proxima vela abrir). Ou seja, no momento da decisao a vela M15 atual
  AINDA NAO FECHOU. Para treinar sem "leakage" (vazamento de futuro), o
  dataset precisa reconstruir exatamente o que era visivel no minuto 13:
  velas M15 fechadas + a vela atual PARCIAL (13 velas de 1m).

Uso:
    python fetch_data.py --days 730 --out data/btcusdt_1m.parquet

Rode na SUA maquina (precisa de acesso a api.binance.com).
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.binance.us/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000  # maximo por request

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    frames = []
    cursor = start_ms
    session = requests.Session()
    while cursor < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        resp = session.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        df = pd.DataFrame(batch, columns=COLS)
        frames.append(df)
        cursor = int(batch[-1][0]) + 60_000  # proximo minuto
        done = (cursor - start_ms) / (end_ms - start_ms) * 100
        print(f"\r{done:5.1f}%  ate {pd.to_datetime(cursor, unit='ms', utc=True)}", end="")
        time.sleep(0.15)  # respeitar rate limit
    print()
    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730, help="quantos dias de historico")
    ap.add_argument("--out", type=str, default="data/btcusdt_1m.parquet")
    args = ap.parse_args()

    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=args.days)

    df = fetch_klines(int(start.timestamp() * 1000), int(end.timestamp() * 1000))

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    df = df[["open_time", "open", "high", "low", "close", "volume",
             "n_trades", "taker_buy_base"]]
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Salvo: {out_path}  ({len(df):,} velas de 1m, "
          f"{df.open_time.min()} -> {df.open_time.max()})")


if __name__ == "__main__":
    main()

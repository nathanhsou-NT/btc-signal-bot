"""
predict_live.py
---------------
Roda em producao: no MINUTO 13 de cada vela M15 (2 min antes da proxima
abrir), busca os candles 1m recentes na Binance, reconstroi as features
exatamente como no treino e imprime o sinal.

A saida pode ser plugada direto no seu bot do Telegram — a funcao
`get_signal()` retorna um dict pronto.

Uso (uma previsao agora):
    python predict_live.py --model model.joblib --payout 0.87

Uso (loop continuo, dispara todo minuto 13/28/43/58):
    python predict_live.py --model model.joblib --payout 0.87 --loop
"""

import argparse
import time

import joblib
import pandas as pd
import requests

from build_dataset import build, DECISION_MINUTES

BASE_URL = "https://api.binance.us/api/v3/klines"
LOOKBACK_DAYS = 5  # suficiente p/ EMA200 M15 + rolling(96)


def fetch_recent_1m(days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    frames, cursor = [], int(start.timestamp() * 1000)
    while cursor < int(end.timestamp() * 1000):
        r = requests.get(BASE_URL, params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": cursor, "limit": 1000}, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        cursor = int(batch[-1][0]) + 60_000
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["open_time", "open", "high", "low", "close", "volume",
                  "close_time", "qv", "n_trades", "taker_buy_base", "tq", "ig"]
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    return df[["open_time", "open", "high", "low", "close",
               "volume", "n_trades", "taker_buy_base"]]


def get_signal(bundle: dict, payout: float) -> dict:
    df1m = fetch_recent_1m()
    ds = build(df1m, live=True)
    row = ds.iloc[[-1]]  # vela atual (parcial)

    X = row[bundle["features"]].astype(float).values
    p_raw = bundle["model"].predict_proba(X)[0, 1]
    p = float(bundle["iso"].predict([p_raw])[0])

    breakeven = 1 / (1 + payout)
    if p >= breakeven:
        direction, conf = "CALL", p
    elif p <= 1 - breakeven:
        direction, conf = "PUT", 1 - p
    else:
        direction, conf = "SKIP", max(p, 1 - p)

    return {
        "candle_atual": str(row["candle_open"].iloc[0]),
        "proxima_vela": str(pd.Timestamp(row["candle_open"].iloc[0])
                            + pd.Timedelta(minutes=15)),
        "p_verde_calibrada": round(p, 4),
        "sinal": direction,
        "confianca": round(conf, 4),
        "breakeven": round(breakeven, 4),
    }


def seconds_until_next_decision() -> float:
    now = pd.Timestamp.now(tz="UTC")
    candle_open = now.floor("15min")
    target = candle_open + pd.Timedelta(minutes=DECISION_MINUTES, seconds=5)
    if now >= target:
        target += pd.Timedelta(minutes=15)
    return (target - now).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--payout", type=float, default=0.87)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(args.model)

    if not args.loop:
        print(get_signal(bundle, args.payout))
        return

    print(f"Loop iniciado — decisao no minuto {DECISION_MINUTES} de cada vela M15.")
    while True:
        wait = seconds_until_next_decision()
        print(f"Aguardando {wait:.0f}s...")
        time.sleep(wait)
        try:
            sig = get_signal(bundle, args.payout)
            print(sig)
            # >>> aqui voce chama o envio pro Telegram / seu automatizador <<<
        except Exception as e:
            print(f"Erro: {e}")


if __name__ == "__main__":
    main()

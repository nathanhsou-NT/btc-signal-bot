"""
diagnose.py
-----------
Investiga por que os sinais estao 100% PUT. Testa 3 hipoteses:

  A) Mercado em queda -> qual o % de velas verdes no periodo recente?
  B) Vies de calibracao -> distribuicao das probabilidades p_verde
     (se tudo fica entre 0.30-0.50, CALL nunca dispara)
  C) Bug de train/serve skew -> compara as features geradas em modo
     "live" (vela parcial) com as do dataset offline p/ a MESMA vela.
     Se divergirem, ha bug na pipeline.

E o mais importante: refaz as decisoes dos ultimos N dias e mostra
quantos CALL/PUT teriam saido e o WINRATE de cada um.

Uso:
    python3 diagnose.py --model model.joblib --days 10 --payout 0.87
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import requests

from build_dataset import build, DECISION_MINUTES

BASE_URL = "https://api.binance.us/api/v3/klines"
WARMUP_DAYS = 4  # p/ EMA200 M15 e rolling(96)


def fetch_1m(days: int) -> pd.DataFrame:
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
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"),
                                     unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    return df[["open_time", "open", "high", "low", "close",
               "volume", "n_trades", "taker_buy_base"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--payout", type=float, default=0.87)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    thr = 1 / (1 + args.payout)

    print(f"Baixando ultimos {args.days + WARMUP_DAYS} dias de 1m...")
    df1m = fetch_1m(args.days + WARMUP_DAYS)

    ds = build(df1m)  # com labels (velas ja fechadas)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
    recent = ds[pd.to_datetime(ds["candle_open"]) >= cutoff].copy()
    print(f"{len(recent)} decisoes reconstruidas nos ultimos {args.days} dias\n")

    # ---------- HIPOTESE A: regime de mercado ----------
    base = recent["y"].mean()
    first_close = df1m[df1m.open_time >= cutoff]["close"].iloc[0]
    last_close = df1m["close"].iloc[-1]
    drift = (last_close / first_close - 1) * 100
    print("=" * 60)
    print("A) REGIME DE MERCADO")
    print(f"   Velas verdes no periodo : {base:.1%}  (50% = neutro)")
    print(f"   Variacao do BTC no periodo: {drift:+.2f}%")

    # ---------- HIPOTESE B: distribuicao de probabilidades ----------
    X = recent[bundle["features"]].astype(float).values
    p_raw = bundle["model"].predict_proba(X)[:, 1]
    p = bundle["iso"].predict(p_raw)
    recent["p"] = p

    print("=" * 60)
    print("B) DISTRIBUICAO DE p_verde (calibrada)")
    q = np.percentile(p, [0, 10, 25, 50, 75, 90, 100])
    print(f"   min={q[0]:.3f}  p10={q[1]:.3f}  p25={q[2]:.3f}  "
          f"mediana={q[3]:.3f}  p75={q[4]:.3f}  p90={q[5]:.3f}  max={q[6]:.3f}")
    print(f"   media={p.mean():.3f}")
    print(f"   Threshold CALL: p >= {thr:.4f}  |  PUT: p <= {1-thr:.4f}")
    pct_call_zone = (p >= thr).mean()
    pct_put_zone = (p <= 1 - thr).mean()
    print(f"   % do tempo na zona CALL: {pct_call_zone:.1%} | zona PUT: {pct_put_zone:.1%}")
    print(f"   p_raw (sem calibracao): media={p_raw.mean():.3f} "
          f"min={p_raw.min():.3f} max={p_raw.max():.3f}")

    # ---------- Winrate real dos sinais que teriam saido ----------
    print("=" * 60)
    print("C) SINAIS QUE TERIAM SAIDO E SEUS RESULTADOS")
    call = recent[recent["p"] >= thr]
    put = recent[recent["p"] <= 1 - thr]
    breakeven = thr
    for nome, grupo, alvo in [("CALL", call, 1), ("PUT", put, 0)]:
        if len(grupo) == 0:
            print(f"   {nome}: 0 sinais")
            continue
        wr = (grupo["y"] == alvo).mean()
        ev = wr * args.payout - (1 - wr)
        print(f"   {nome}: {len(grupo)} sinais | winrate={wr:.1%} "
              f"(breakeven {breakeven:.1%}) | EV/trade={ev:+.4f}")

    # ---------- HIPOTESE D: train/serve skew ----------
    print("=" * 60)
    print("D) TESTE DE TRAIN/SERVE SKEW (features live vs offline)")
    # escolhe uma vela do meio do periodo e simula o minuto 13 dela
    test_row = recent.iloc[len(recent) // 2]
    t_open = pd.Timestamp(test_row["candle_open"])
    cut = t_open + pd.Timedelta(minutes=DECISION_MINUTES)
    df_cut = df1m[df1m["open_time"] < cut]
    ds_live = build(df_cut, live=True)
    live_row = ds_live[ds_live["candle_open"] == t_open]
    if len(live_row) == 0:
        print("   ERRO: vela nao encontrada em modo live — ISSO E UM BUG.")
    else:
        feats = bundle["features"]
        off = test_row[feats].astype(float).values
        liv = live_row[feats].astype(float).values[0]
        diff = np.abs(off - liv)
        # ignora NaN==NaN
        mask = ~(np.isnan(off) & np.isnan(liv))
        maxdiff = np.nanmax(diff[mask]) if mask.any() else 0.0
        print(f"   Vela testada: {t_open}")
        print(f"   Diferenca maxima entre features live vs offline: {maxdiff:.2e}")
        if maxdiff > 1e-6:
            worst = np.array(feats)[mask][np.nanargmax(diff[mask])]
            print(f"   >>> DIVERGENCIA na feature '{worst}' — ha skew na pipeline!")
        else:
            print("   OK — pipeline live identica ao treino, sem skew.")

    # ---------- Veredito ----------
    print("=" * 60)
    print("COMO LER:")
    print(" - Se (A) mostra <45% de velas verdes e queda forte: o mercado")
    print("   realmente esta bearish; PUTs dominarem e esperado. O que importa")
    print("   e o winrate em (C): PUT acima do breakeven = funcionando.")
    print(" - Se (B) mostra p_verde SEMPRE abaixo de ~0.50 mesmo com mercado")
    print("   lateral: vies de calibracao -> retreine com --calib-days 28.")
    print(" - Se (D) acusa divergencia: bug de pipeline, me mande a saida.")


if __name__ == "__main__":
    main()

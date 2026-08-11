"""
build_dataset.py
----------------
Constroi o dataset supervisionado a partir dos candles de 1 minuto,
respeitando a restricao operacional da corretora:

  A DECISAO ACONTECE NO MINUTO 13 DA VELA M15 ATUAL (vela N).
  O ALVO E A COR DA PROXIMA VELA (vela N+1).

Entao, para cada vela N, o que e "visivel" no momento da decisao:
  1. Todas as velas M15 ate N-1 (fechadas)         -> indicadores com shift(1)
  2. A vela N PARCIAL: minutos 0..12 (13 velas 1m)  -> features parciais
  3. NADA da vela N completa, NADA da vela N+1

Label:
  y = 1 se close(N+1) > open(N+1)  (vela verde / CALL)
  y = 0 se close(N+1) < open(N+1)  (vela vermelha / PUT)
  Dojis exatos (close == open) sao descartados do treino.

Uso:
    python build_dataset.py --in data/btcusdt_1m.parquet --out data/dataset_m15.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DECISION_MINUTES = 13  # minutos da vela atual visiveis na hora da decisao


# ----------------------------------------------------------------------
# Indicadores classicos (calculados sobre M15 FECHADO, depois shift(1))
# ----------------------------------------------------------------------

def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA), como no TradingView."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = rma(delta.clip(lower=0), period)
    loss = rma(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, period)


# ----------------------------------------------------------------------
# Construcao
# ----------------------------------------------------------------------

def build(df1m: pd.DataFrame, live: bool = False) -> pd.DataFrame:
    """live=True: mantem a ultima vela mesmo sem label (para previsao em producao)."""
    df1m = df1m.sort_values("open_time").reset_index(drop=True)
    df1m = df1m.set_index("open_time")

    # ---- 1) M15 fechado -------------------------------------------------
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "taker_buy_base": "sum", "n_trades": "sum",
    }
    m15 = df1m.resample("15min").agg(agg).dropna(subset=["open"])

    # descarta velas M15 incompletas. Em live, SO a ultima vela (a atual)
    # pode estar parcial (>=13 min); as historicas precisam de 15 min,
    # senao uma primeira vela parcial contamina o warm-up das EMAs longas.
    counts = df1m["close"].resample("15min").count().reindex(m15.index)
    if live:
        is_last = m15.index == m15.index.max()
        m15 = m15[(counts >= 15) | (is_last & (counts >= DECISION_MINUTES))]
    else:
        m15 = m15[counts >= 15]

    c, h, l, o, v = m15["close"], m15["high"], m15["low"], m15["open"], m15["volume"]

    feat = pd.DataFrame(index=m15.index)

    # retornos log de velas fechadas
    logret = np.log(c / c.shift(1))
    for k in [1, 2, 3, 4, 8, 16, 32, 96]:
        feat[f"ret_{k}"] = np.log(c / c.shift(k))

    # volatilidade realizada
    feat["vol_8"] = logret.rolling(8).std()
    feat["vol_32"] = logret.rolling(32).std()
    feat["vol_ratio"] = feat["vol_8"] / feat["vol_32"]

    # RSI / ATR (Wilder)
    feat["rsi_14"] = rsi_wilder(c, 14)
    feat["rsi_7"] = rsi_wilder(c, 7)
    atr = atr_wilder(m15, 14)
    feat["atr_norm"] = atr / c  # ATR normalizado pelo preco

    # medias moveis normalizadas (diferenca em unidades de ATR)
    for p in [9, 21, 50, 200]:
        ema = c.ewm(span=p, adjust=False).mean()
        feat[f"ema{p}_dist"] = (c - ema) / atr
    feat["ema9_21"] = (c.ewm(span=9, adjust=False).mean()
                       - c.ewm(span=21, adjust=False).mean()) / atr
    feat["ema50_200"] = (c.ewm(span=50, adjust=False).mean()
                         - c.ewm(span=200, adjust=False).mean()) / atr

    # anatomia da ultima vela fechada
    rng = (h - l).replace(0, np.nan)
    feat["body_pct"] = (c - o) / rng
    feat["upper_wick"] = (h - np.maximum(c, o)) / rng
    feat["lower_wick"] = (np.minimum(c, o) - l) / rng
    feat["close_pos"] = (c - l) / rng  # onde fechou dentro do range

    # sequencia de cores (streak)
    color = np.sign(c - o)
    streak = color.groupby((color != color.shift()).cumsum()).cumcount() + 1
    feat["streak"] = (streak * color).astype(float)

    # volume relativo e fluxo comprador
    feat["vol_z"] = (v - v.rolling(96).mean()) / v.rolling(96).std()
    feat["taker_ratio"] = m15["taker_buy_base"] / v.replace(0, np.nan)

    # tendencia de timeframe maior (H1 e H4 sintetizados do M15 fechado)
    for tf, k in [("h1", 4), ("h4", 16)]:
        c_tf = c.rolling(k).apply(lambda x: x.iloc[-1], raw=False)  # close alinhado
        ema_fast = c.ewm(span=9 * k, adjust=False).mean()
        ema_slow = c.ewm(span=21 * k, adjust=False).mean()
        feat[f"{tf}_trend"] = (ema_fast - ema_slow) / atr
        feat[f"{tf}_ret"] = np.log(c / c.shift(k))

    # >>> TUDO acima usa a vela M15 do proprio indice. Na decisao da vela N,
    # a ultima FECHADA e a N-1 -> shift(1) em bloco elimina o leakage. <<<
    feat = feat.shift(1)

    # ---- 2) Vela N parcial (minutos 0..12) -------------------------------
    # agrupa cada minuto pela vela M15 a que pertence
    grp = df1m.groupby(pd.Grouper(freq="15min"))
    first13 = df1m.groupby(pd.Grouper(freq="15min")).head(0)  # placeholder

    def partial_agg(g: pd.DataFrame) -> pd.Series:
        g13 = g.iloc[:DECISION_MINUTES]
        if len(g13) < DECISION_MINUTES:
            return pd.Series(dtype=float)
        p_open = g13["open"].iloc[0]
        p_close = g13["close"].iloc[-1]
        p_high = g13["high"].max()
        p_low = g13["low"].min()
        p_vol = g13["volume"].sum()
        p_taker = g13["taker_buy_base"].sum()
        # micro-momentum dentro da vela parcial
        last3 = np.log(g13["close"].iloc[-1] / g13["close"].iloc[-4]) if len(g13) >= 4 else np.nan
        return pd.Series({
            "p_open": p_open, "p_close": p_close, "p_high": p_high,
            "p_low": p_low, "p_vol": p_vol, "p_taker": p_taker,
            "p_last3_mom": last3,
        })

    partial = grp.apply(partial_agg)
    if isinstance(partial, pd.Series):
        partial = partial.unstack()
    partial = partial.reindex(m15.index)

    prng = (partial["p_high"] - partial["p_low"]).replace(0, np.nan)
    feat["p_body_atr"] = (partial["p_close"] - partial["p_open"]) / atr.shift(1)
    feat["p_body_pct"] = (partial["p_close"] - partial["p_open"]) / prng
    feat["p_close_pos"] = (partial["p_close"] - partial["p_low"]) / prng
    feat["p_range_atr"] = prng / atr.shift(1)
    feat["p_last3_mom"] = partial["p_last3_mom"]
    feat["p_taker_ratio"] = partial["p_taker"] / partial["p_vol"].replace(0, np.nan)
    # volume parcial vs volume tipico dos primeiros 13 min (proxy: media movel)
    feat["p_vol_z"] = ((partial["p_vol"] - partial["p_vol"].rolling(96).mean().shift(1))
                       / partial["p_vol"].rolling(96).std().shift(1))

    # ---- 3) Sazonalidade -------------------------------------------------
    idx = feat.index
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # ---- 4) Label: cor da vela N+1 ---------------------------------------
    next_open = o.shift(-1)
    next_close = c.shift(-1)
    label = pd.Series(np.where(next_close > next_open, 1,
                      np.where(next_close < next_open, 0, np.nan)),
                      index=m15.index, name="y")

    ds = feat.copy()
    ds["y"] = label
    ds["decision_time"] = ds.index + pd.Timedelta(minutes=DECISION_MINUTES)
    if not live:
        ds = ds.dropna(subset=["y"])
    # remove o warm-up dos indicadores
    ds = ds.dropna(thresh=int(0.9 * (ds.shape[1] - 2)))
    if not live:
        ds["y"] = ds["y"].astype(int)
    return ds.reset_index().rename(columns={"open_time": "candle_open"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/btcusdt_1m.parquet")
    ap.add_argument("--out", default="data/dataset_m15.parquet")
    args = ap.parse_args()

    df1m = pd.read_parquet(args.inp)
    ds = build(df1m)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)

    n_feat = ds.shape[1] - 3  # menos candle_open, decision_time, y
    bal = ds["y"].mean()
    print(f"Dataset: {len(ds):,} amostras | {n_feat} features | "
          f"balanco (verde) = {bal:.3f}")
    print(f"Periodo: {ds.candle_open.min()} -> {ds.candle_open.max()}")
    print(f"Salvo em {out}")


if __name__ == "__main__":
    main()

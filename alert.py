"""
=============================================================
  BTC M15 — Alerta Automático no Telegram
  Baseado no indicador ML XGBoost do TradingView
  + Filtro de tendência macro (corrige o excesso de CALL em queda)
=============================================================
"""

import ccxt
import pandas as pd
import schedule
import time
import asyncio
import os
from datetime import datetime, timezone, timedelta
from telegram import Bot

# ============================================================
# ⚙️ CONFIGURAÇÃO — PREENCHA AQUI
# ============================================================

# Token e chat id vêm de variáveis de ambiente (nunca hardcoded aqui,
# esse arquivo vai pro GitHub). Defina TELEGRAM_TOKEN e TELEGRAM_CHAT_ID
# no seu start_bot.bat local antes de rodar.
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID        = os.environ.get("TELEGRAM_CHAT_ID", "")
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise SystemExit(
        "Defina as variaveis de ambiente TELEGRAM_TOKEN e TELEGRAM_CHAT_ID "
        "antes de rodar (veja start_bot.bat)."
    )

SYMBOL    = "BTC/USD"
TIMEFRAME = "15m"

# Fuso horário (Florida = UTC-4 no verão, UTC-5 no inverno)
UTC_OFFSET = -4

# ---- 📈 FILTRO DE TENDÊNCIA MACRO --------------------------
# Mede a direção num timeframe maior para evitar operar contra o "dia".
TREND_TIMEFRAME  = "1h"    # timeframe da tendência (1h ou 4h funcionam bem)
TREND_EMA_PERIOD = 50      # EMA usada para definir a tendência
TREND_SLOPE_BARS = 10      # nº de barras para medir a inclinação da EMA
SLOPE_THRESHOLD  = 0.05    # inclinação mínima (%) p/ considerar "em tendência"

# Modo do filtro:
#   "moderado" -> bloqueia só o claramente contra a tendência
#                 (CALL em QUEDA, PUT em ALTA). NEUTRO libera os dois.
#   "estrito"  -> só opera A FAVOR da tendência
#                 (CALL só em ALTA, PUT só em QUEDA). NEUTRO bloqueia tudo.
TREND_FILTER_MODE = "moderado"

# ============================================================
# 📡 BUSCA DE DADOS
# ============================================================

exchange = ccxt.coinbase()

def get_ohlcv(timeframe=TIMEFRAME, limit=300):
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Converte para horário local (Florida)
    df["timestamp"] = df["timestamp"].dt.tz_convert(timezone(timedelta(hours=UTC_OFFSET)))
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

# ============================================================
# 📐 FEATURES
# ============================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calc_features(df):
    ema9 = calc_ema(df["close"], 9)
    df["price_vs_ema9_pct"] = (df["close"] - ema9) / ema9 * 100.0
    df["rsi_7"] = calc_rsi(df["close"], 7)
    df["is_bull"] = df["close"] > df["open"]
    consec = []
    count = 0
    for bull in df["is_bull"]:
        count = count + 1 if bull else 0
        consec.append(count)
    df["consec_bull"] = consec
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    return df

# ============================================================
# 🗂️ DISCRETIZAÇÃO
# ============================================================

def bin_f1(v):
    if v < -0.169065: return 0
    if v < -0.038055: return 1
    if v <  0.048559: return 2
    if v <  0.179754: return 3
    return 4

def bin_f2(v):
    if v < 37.6598: return 0
    if v < 46.6291: return 1
    if v < 54.2769: return 2
    if v < 63.0472: return 3
    return 4

def bin_f3(v):
    if v < 1: return 0
    if v < 2: return 1
    return 2

def bin_f4(v):
    if v <  4.14: return 0
    if v < 13.15: return 1
    if v < 27.29: return 2
    if v < 55.60: return 3
    return 4

# ============================================================
# 🧠 LOOKUP — 28 REGRAS
# ============================================================

RULES = {
    (0, 0, 0, 2): 0.5586,
    (0, 0, 0, 3): 0.5580,
    (0, 0, 0, 1): 0.5565,
    (1, 0, 0, 2): 0.5536,
    (0, 0, 0, 0): 0.5504,
    (4, 3, 2, 2): 0.4472,
    (4, 4, 0, 2): 0.4465,
    (4, 3, 1, 1): 0.4430,
    (3, 4, 1, 2): 0.4409,
    (4, 3, 2, 1): 0.4402,
    (3, 4, 2, 2): 0.4401,
    (4, 3, 1, 0): 0.4391,
    (4, 3, 2, 0): 0.4379,
    (3, 4, 0, 1): 0.4361,
    (4, 4, 0, 1): 0.4347,
    (4, 4, 0, 0): 0.4330,
    (2, 4, 0, 0): 0.4330,
    (3, 4, 0, 0): 0.4286,
    (4, 4, 2, 2): 0.4284,
    (4, 4, 1, 2): 0.4273,
    (3, 4, 1, 1): 0.4225,
    (3, 4, 2, 1): 0.4193,
    (4, 4, 2, 1): 0.4174,
    (4, 4, 2, 0): 0.4158,
    (4, 4, 1, 1): 0.4156,
    (4, 4, 1, 0): 0.4152,
    (3, 4, 1, 0): 0.4137,
    (3, 4, 2, 0): 0.4122,
}

def classify(row):
    try:
        key = (
            bin_f1(float(row["price_vs_ema9_pct"])),
            bin_f2(float(row["rsi_7"])),
            bin_f3(float(row["consec_bull"])),
            bin_f4(float(row["upper_wick"]))
        )
        prob_bull = RULES.get(key, 0.5)
        if prob_bull >= 0.55:
            return "CALL", prob_bull
        elif prob_bull <= 0.45:
            return "PUT", prob_bull
        return None, 0.5
    except Exception as e:
        print(f"[ERRO classify] {e}")
        return None, 0.5

# ============================================================
# 📈 TENDÊNCIA MACRO
# ============================================================

def get_trend():
    """Mede a tendência num timeframe maior (default 1h).
    Retorna ('ALTA' | 'QUEDA' | 'NEUTRO', inclinacao_pct)."""
    need = TREND_EMA_PERIOD + TREND_SLOPE_BARS + 20
    df   = get_ohlcv(timeframe=TREND_TIMEFRAME, limit=need)
    ema  = calc_ema(df["close"], TREND_EMA_PERIOD)

    price    = float(df["close"].iloc[-1])
    ema_now  = float(ema.iloc[-1])
    ema_prev = float(ema.iloc[-1 - TREND_SLOPE_BARS])
    slope    = (ema_now - ema_prev) / ema_prev * 100.0

    if price > ema_now and slope >  SLOPE_THRESHOLD:
        return "ALTA", slope
    if price < ema_now and slope < -SLOPE_THRESHOLD:
        return "QUEDA", slope
    return "NEUTRO", slope

def trend_allows(signal, trend):
    """Decide se o sinal pode passar, dado o filtro de tendência."""
    if TREND_FILTER_MODE == "estrito":
        if signal == "CALL":
            return trend == "ALTA"
        if signal == "PUT":
            return trend == "QUEDA"
        return False
    # modo "moderado" (padrão)
    if signal == "CALL" and trend == "QUEDA":
        return False
    if signal == "PUT" and trend == "ALTA":
        return False
    return True

# ============================================================
# 📲 ENVIO DO TELEGRAM
# ============================================================

async def send_telegram(message):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")

def notify(signal, prob, price, current_candle_time, trend, slope):
    # Horário da vela de ENTRADA (próxima vela = +15 min)
    entry_candle = current_candle_time + timedelta(minutes=15)
    entry_str = entry_candle.strftime("%H:%M")
    date_str  = entry_candle.strftime("%d/%m/%Y")

    if signal == "CALL":
        emoji     = "🟢"
        prob_pct  = round(prob * 100)
        direction = "CALL (COMPRA)"
    else:
        emoji     = "🔴"
        prob_pct  = round((1 - prob) * 100)
        direction = "PUT (VENDA)"

    trend_emoji = {"ALTA": "📈", "QUEDA": "📉", "NEUTRO": "➡️"}[trend]

    msg = (
        f"{emoji} <b>SINAL ML — {direction}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 Ativo: <b>BTC/USD M15</b>\n"
        f"💰 Preço atual: <b>${price:,.2f}</b>\n"
        f"🎯 Probabilidade: <b>{prob_pct}%</b>\n"
        f"{trend_emoji} Tendência ({TREND_TIMEFRAME}): <b>{trend}</b> ({slope:+.2f}%)\n"
        f"🕯️ Entrar na vela: <b>{entry_str} ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ <i>Não é garantia de resultado</i>"
    )
    asyncio.run(send_telegram(msg))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Alerta enviado: {direction} "
          f"| Tend: {trend} | Entrada: {entry_str} | Prob: {prob_pct}%")

# ============================================================
# 🔄 VERIFICAÇÃO PRINCIPAL
# ============================================================

last_signal_bar = None

def check_signal():
    global last_signal_bar
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Verificando sinal...")
        df = get_ohlcv()
        df = calc_features(df)

        row = df.iloc[-1]
        signal, prob = classify(row)

        bar_id = str(row["timestamp"])

        if signal and bar_id != last_signal_bar:
            trend, slope = get_trend()
            if trend_allows(signal, trend):
                last_signal_bar = bar_id
                notify(signal, prob, row["close"], row["timestamp"], trend, slope)
            else:
                last_signal_bar = bar_id  # marca a vela p/ não reavaliar
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Sinal {signal} BLOQUEADO pela tendência {trend} ({slope:+.2f}%).")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sem sinal nesta vela.")

    except Exception as e:
        print(f"[ERRO] {e}")

# ============================================================
# ▶️ EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  BTC M15 — Monitor de Sinais ML (com filtro de tendência)")
    print(f"  Filtro: {TREND_FILTER_MODE} | Tendência: {TREND_TIMEFRAME} EMA{TREND_EMA_PERIOD}")
    print("  Verificando a cada 15 minutos...")
    print("=" * 50)

    check_signal()

    # 2 minutos antes do fechamento de cada vela M15
    for h in range(24):
        for m in [13, 28, 43, 58]:
            schedule.every().day.at(f"{h:02d}:{m:02d}").do(check_signal)

    while True:
        schedule.run_pending()
        time.sleep(30)

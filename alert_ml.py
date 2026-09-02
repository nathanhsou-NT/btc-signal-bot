"""
=============================================================
  BTC M15 — Alerta Automático no Telegram  (v2 — modelo real)
  Substitui a tabela lookup de 28 regras pelo XGBoost treinado
  com walk-forward + calibração (model.joblib).

  Mantém: mesmos horários (:13/:28/:43/:58), mesma mensagem
  HTML, mesmo bot/canal — seu listener não percebe diferença.

  Requisitos (mesma pasta): build_dataset.py + model.joblib
  pip install pandas numpy requests xgboost scikit-learn joblib \
              schedule python-telegram-bot
=============================================================
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import requests
import schedule
from telegram import Bot

from build_dataset import build  # mesma engenharia de features do treino

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

MODEL_PATH = "model.joblib"

# ---- MODO DO SINAL --------------------------------------------
# "quantile"  -> dispara nas velas em que o modelo esta RELATIVAMENTE
#                mais convicto (ranking vs ultimos dias). Garante volume
#                de sinais configuravel.
# "breakeven" -> dispara sempre que p passar de MIN_PROB (ou do breakeven
#                do payout, se MIN_PROB for None). Threshold fixo, direto
#                comparavel com a tabela do report.json.
#
# Testado no walk-forward (report.json): thr=0.53 -> ~36 sinais/dia,
# winrate 54,4% (breakeven 53,48%), EV +0,017/trade. Ajuste MIN_PROB pra
# subir/descer o volume de sinais (0.55 ~17/dia, 0.58 ~6/dia, 0.60 ~3/dia).
SIGNAL_MODE = "breakeven"
MIN_PROB = 0.53

TARGET_SIGNALS_PER_DAY = 8   # so usado se SIGNAL_MODE = "quantile"
QUANTILE_LOOKBACK_DAYS = 4   # janela p/ calcular o ranking (modo quantile)

PAYOUT = 0.90                # payout real da corretora, exibido na msg

# Fuso horário para exibir o horário da vela na mensagem
UTC_OFFSET = -4          # Florida (verão)

# ============================================================
# 📡 DADOS (Binance 1m — MESMA fonte usada no treino)
# ============================================================

BINANCE_URL = "https://api.binance.us/api/v3/klines"
LOOKBACK_DAYS = 5  # warm-up p/ EMA200 M15 + rolling(96)


def fetch_recent_1m(days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=days)
    frames, cursor = [], int(start.timestamp() * 1000)
    while cursor < int(end.timestamp() * 1000):
        r = requests.get(BINANCE_URL, params={
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

# ============================================================
# 🧠 CLASSIFICAÇÃO (substitui a tabela RULES)
# ============================================================

BUNDLE = joblib.load(MODEL_PATH)
THRESHOLD = MIN_PROB if MIN_PROB is not None else 1 / (1 + PAYOUT)


def classify():
    """Retorna (signal, prob_bull, price, candle_open_utc, extra_info)."""
    df1m = fetch_recent_1m()
    ds = build(df1m, live=True)

    # probabilidade de TODAS as velas recentes (p/ ranking) + da atual
    X_all = ds[BUNDLE["features"]].astype(float).values
    p_all = BUNDLE["iso"].predict(BUNDLE["model"].predict_proba(X_all)[:, 1])
    prob_bull = float(p_all[-1])

    price = float(df1m["close"].iloc[-1])
    candle_open = pd.Timestamp(ds["candle_open"].iloc[-1])

    if SIGNAL_MODE == "quantile":
        n_ref = QUANTILE_LOOKBACK_DAYS * 96
        ref = p_all[:-1][-n_ref:]              # historico, sem a vela atual
        frac = TARGET_SIGNALS_PER_DAY / 96 / 2  # metade CALL, metade PUT
        call_thr = float(pd.Series(ref).quantile(1 - frac))
        put_thr = float(pd.Series(ref).quantile(frac))
        info = f"ranking: CALL>= {call_thr:.4f} | PUT<= {put_thr:.4f}"
        if prob_bull >= call_thr:
            return "CALL", prob_bull, price, candle_open, info
        if prob_bull <= put_thr:
            return "PUT", prob_bull, price, candle_open, info
        return None, prob_bull, price, candle_open, info

    # modo breakeven (original)
    info = f"threshold: {THRESHOLD:.4f}"
    if prob_bull >= THRESHOLD:
        return "CALL", prob_bull, price, candle_open, info
    if prob_bull <= 1 - THRESHOLD:
        return "PUT", prob_bull, price, candle_open, info
    return None, prob_bull, price, candle_open, info

# ============================================================
# 📲 ENVIO DO TELEGRAM (formato idêntico ao anterior)
# ============================================================

async def send_telegram(message):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")


def notify(signal, prob, price, candle_open_utc):
    local_tz = timezone(timedelta(hours=UTC_OFFSET))
    entry_candle = (candle_open_utc + timedelta(minutes=15)).astimezone(local_tz)
    entry_str = entry_candle.strftime("%H:%M")
    date_str = entry_candle.strftime("%d/%m/%Y")

    if signal == "CALL":
        emoji = "🟢"
        prob_pct = round(prob * 100)
        direction = "CALL (COMPRA)"
    else:
        emoji = "🔴"
        prob_pct = round((1 - prob) * 100)
        direction = "PUT (VENDA)"

    msg = (
        f"{emoji} <b>SINAL ML — {direction}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 Ativo: <b>BTC/USD M15</b>\n"
        f"💰 Preço atual: <b>${price:,.2f}</b>\n"
        f"🎯 Probabilidade: <b>{prob_pct}%</b>\n"
        f"🕯️ Entrar na vela: <b>{entry_str} ({date_str})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ <i>Não é garantia de resultado</i>"
    )
    asyncio.run(send_telegram(msg))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Alerta enviado: {direction} "
          f"| Entrada: {entry_str} | Prob: {prob_pct}%")

# ============================================================
# 🔄 VERIFICAÇÃO PRINCIPAL
# ============================================================

last_signal_bar = None


def check_signal():
    global last_signal_bar
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Verificando sinal...")
        signal, prob, price, candle_open, info = classify()
        bar_id = str(candle_open)

        if signal and bar_id != last_signal_bar:
            last_signal_bar = bar_id
            notify(signal, prob, price, candle_open)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Sem sinal nesta vela (p_verde={prob:.4f} | {info}).")

    except Exception as e:
        print(f"[ERRO] {e}")

# ============================================================
# ▶️ EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  BTC M15 — Monitor de Sinais ML v2 (modelo walk-forward)")
    print(f"  Modelo: {MODEL_PATH} | Modo: {SIGNAL_MODE} "
          f"| Alvo: ~{TARGET_SIGNALS_PER_DAY} sinais/dia")
    print("  Decisão no minuto 13/28/43/58 (2 min antes da próxima vela)")
    print("=" * 50)

    check_signal()

    for h in range(24):
        for m in [13, 28, 43, 58]:
            schedule.every().day.at(f"{h:02d}:{m:02d}").do(check_signal)

    while True:
        schedule.run_pending()
        time.sleep(30)

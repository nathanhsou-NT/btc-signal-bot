"""
notify_retrain.py
------------------
Le report.json (gerado pelo train_walkforward.py) e manda um resumo do
retreino semanal pro Telegram. Usado pelo workflow do GitHub Actions
(.github/workflows/weekly-retrain.yml), mas tambem pode ser rodado a mao.

Variaveis de ambiente necessarias:
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

Uso:
    python notify_retrain.py
"""

import json
import os
from pathlib import Path

import requests

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def build_message() -> str:
    report_path = Path("report.json")
    if not report_path.exists():
        return "⚠️ <b>Retreino semanal falhou</b>\nreport.json não foi gerado."

    r = json.loads(report_path.read_text())
    breakeven = r["breakeven_winrate"]

    linhas = [
        "🔄 <b>Retreino semanal concluído</b>",
        f"Amostras OOS: {r['n_oos']:,} ({r['folds']} folds)",
        f"AUC: {r['auc']:.4f}",
        f"Breakeven (payout {r['payout']:.0%}): {breakeven:.2%}",
        "",
        "<b>thr | cobertura | winrate | EV/trade</b>",
    ]
    for t in r["thresholds"]:
        flag = " ✅" if t["ev_por_trade"] > 0 else ""
        linhas.append(
            f"{t['threshold']:.2f} | {t['cobertura']:.1%} | "
            f"{t['winrate']:.1%} | {t['ev_por_trade']:+.4f}{flag}"
        )
    return "\n".join(linhas)


def main():
    msg = build_message()
    print(msg)

    if not TOKEN or not CHAT_ID:
        print("\n[aviso] TELEGRAM_TOKEN/TELEGRAM_CHAT_ID não configurados — "
              "notificação não enviada.")
        return

    resp = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
        timeout=15,
    )
    print(f"\n[telegram] status={resp.status_code} resp={resp.text[:200]}")


if __name__ == "__main__":
    main()

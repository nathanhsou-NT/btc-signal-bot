"""
train_walkforward.py
--------------------
Treina e avalia com WALK-FORWARD (a unica validacao honesta em serie temporal):

  |---- treino (rolling) ----|--calib--|-- teste --|
                                        avanca -->

Cada janela de teste so ve modelos treinados com dados anteriores a ela.
As previsoes out-of-sample de todas as janelas sao concatenadas e avaliadas
juntas, incluindo:

  - Acuracia e AUC gerais
  - CALIBRACAO (isotonic) para que p=0.60 signifique ~60% de acerto real
  - Analise de EV vs payout da corretora: com payout de 87%, o breakeven e
        p_min = 1 / (1 + 0.87) = 53.48%
    ou seja: so vale operar quando a probabilidade calibrada passa disso.
  - Curva cobertura x winrate por threshold (opera menos, acerta mais)

Uso:
    python train_walkforward.py --data data/dataset_m15.parquet --payout 0.87
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

META_COLS = ["candle_open", "decision_time", "y"]


from calibration import CenteredPlattCalibrator, PlattCalibrator


def make_calibrator(method: str):
    if method == "sigmoid":
        return PlattCalibrator()
    if method == "centered":
        return CenteredPlattCalibrator()
    return IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)


def make_model(seed: int, n_estimators: int = 600,
               early_stopping: bool = True) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        gamma=0.1,
        eval_metric="logloss",
        early_stopping_rounds=50 if early_stopping else None,
        random_state=seed,
        n_jobs=-1,
    )


def walk_forward(ds: pd.DataFrame, train_days: int, calib_days: int,
                 test_days: int, step_days: int, calib_method: str = "sigmoid"):
    ds = ds.sort_values("candle_open").reset_index(drop=True)
    t = pd.to_datetime(ds["candle_open"])
    feats = [c for c in ds.columns if c not in META_COLS]
    X = ds[feats].astype(float).values
    y = ds["y"].values

    start = t.min() + pd.Timedelta(days=train_days + calib_days)
    end = t.max()

    oos = []
    best_iters = []
    fold = 0
    cursor = start
    while cursor + pd.Timedelta(days=test_days) <= end + pd.Timedelta(days=step_days):
        tr_mask = (t >= cursor - pd.Timedelta(days=train_days + calib_days)) & \
                  (t < cursor - pd.Timedelta(days=calib_days))
        ca_mask = (t >= cursor - pd.Timedelta(days=calib_days)) & (t < cursor)
        te_mask = (t >= cursor) & (t < cursor + pd.Timedelta(days=test_days))

        if tr_mask.sum() < 2000 or ca_mask.sum() < 300 or te_mask.sum() < 100:
            cursor += pd.Timedelta(days=step_days)
            continue

        model = make_model(seed=fold)
        model.fit(X[tr_mask], y[tr_mask],
                  eval_set=[(X[ca_mask], y[ca_mask])], verbose=False)
        best_iters.append(int(model.best_iteration) + 1)

        p_calib_raw = model.predict_proba(X[ca_mask])[:, 1]
        iso = make_calibrator(calib_method)
        iso.fit(p_calib_raw, y[ca_mask])

        p_te_raw = model.predict_proba(X[te_mask])[:, 1]
        p_test = iso.predict(p_te_raw)
        p_test_c = make_calibrator("centered").fit(p_calib_raw, y[ca_mask]).predict(p_te_raw)
        oos.append(pd.DataFrame({
            "time": t[te_mask].values,
            "y": y[te_mask],
            "p": p_test,
            "p_centered": p_test_c,
            "fold": fold,
        }))
        fold += 1
        cursor += pd.Timedelta(days=step_days)

    if not oos:
        raise RuntimeError("Nenhum fold gerado — dataset curto demais para os parametros.")
    return pd.concat(oos, ignore_index=True), feats, model, best_iters


def evaluate(oos: pd.DataFrame, payout: float, col: str = "p") -> dict:
    oos = oos.assign(p=oos[col])
    breakeven = 1 / (1 + payout)
    res = {"n_oos": int(len(oos)),
           "folds": int(oos["fold"].nunique()),
           "base_rate_verde": float(oos["y"].mean()),
           "auc": float(roc_auc_score(oos["y"], oos["p"])),
           "acc_sempre_opera": float(((oos["p"] > 0.5) == oos["y"]).mean()),
           "payout": payout,
           "breakeven_winrate": breakeven,
           "thresholds": []}

    # decide CALL se p > thr, PUT se p < 1-thr, senao fica de fora
    for thr in [0.50, 0.53, breakeven, 0.55, 0.58, 0.60, 0.65]:
        call = oos["p"] >= thr
        put = oos["p"] <= 1 - thr
        traded = call | put
        n = int(traded.sum())
        if n == 0:
            continue
        correct = np.where(call[traded], oos["y"][traded] == 1, oos["y"][traded] == 0)
        winrate = float(correct.mean())
        ev = winrate * payout - (1 - winrate)  # EV por trade (stake=1)
        res["thresholds"].append({
            "threshold": round(float(thr), 4),
            "trades": n,
            "cobertura": round(n / len(oos), 3),
            "winrate": round(winrate, 4),
            "ev_por_trade": round(ev, 4),
            "sinais_dia": round(n / len(oos) * 96, 1),
            "ev_dia": round(ev * n / len(oos) * 96, 3),
            "pct_call": round(float(call[traded].mean()), 3),
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_m15.parquet")
    ap.add_argument("--payout", type=float, default=0.87,
                    help="payout da corretora (0.87 = 87%%)")
    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--calib-days", type=int, default=28)
    ap.add_argument("--calib-method", choices=["auto", "sigmoid", "centered", "isotonic"],
                    default="centered",
                    help="centered (padrao) = CALL/PUT simetricos, sem vies da tendencia "
                         "dos ultimos 28 dias; auto = o de maior EV/dia no walk-forward")
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--report", default="report.json")
    ap.add_argument("--save-model", default=None,
                    help="ex: model.joblib — salva modelo final p/ producao")
    ap.add_argument("--min-signal-thr", type=float, default=0.53,
                    help="threshold usado pelo alert_ml.py (p/ trava de seguranca)")
    ap.add_argument("--min-coverage", type=float, default=0.05,
                    help="cobertura minima do modelo final na janela de calibracao")
    args = ap.parse_args()

    ds = pd.read_parquet(args.data)
    print(f"Dataset: {len(ds):,} amostras")

    oos, feats, last_model, best_iters = walk_forward(
        ds, args.train_days, args.calib_days, args.test_days, args.step_days,
        "isotonic" if args.calib_method == "isotonic" else "sigmoid")

    res = evaluate(oos, args.payout, "p")
    res_c = evaluate(oos, args.payout, "p_centered")

    # escolhe a calibracao com maior EV/dia no threshold usado ao vivo
    def ev_dia(r):
        t = [x for x in r["thresholds"] if abs(x["threshold"] - args.min_signal_thr) < 1e-9]
        return t[0]["ev_dia"] if t else float("-inf")
    if args.calib_method == "auto":
        args.calib_method = "centered" if ev_dia(res_c) >= ev_dia(res) else "sigmoid"
    if args.calib_method == "centered":
        res, res_c = res_c, res
    res["calibracao"] = args.calib_method
    res["alternativa"] = {"calibracao": "sigmoid" if args.calib_method == "centered" else "centered",
                          "thresholds": res_c["thresholds"]}
    print(f"\nCalibracao escolhida: {args.calib_method} "
          f"(EV/dia @{args.min_signal_thr}: {ev_dia(res):+.3f} vs alternativa {ev_dia(res_c):+.3f})")

    print("\n===== RESULTADO OUT-OF-SAMPLE (walk-forward) =====")
    print(f"Amostras OOS : {res['n_oos']:,}  ({res['folds']} folds)")
    print(f"Base rate    : {res['base_rate_verde']:.4f} (vela verde)")
    print(f"AUC          : {res['auc']:.4f}")
    print(f"Acc (thr .50): {res['acc_sempre_opera']:.4f}")
    print(f"\nBreakeven com payout {args.payout:.0%}: winrate > {res['breakeven_winrate']:.2%}")
    print(f"\n{'thr':>6} {'trades':>8} {'cobert.':>8} {'winrate':>8} {'EV/trade':>9}")
    for r in res["thresholds"]:
        flag = " <-- LUCRATIVO" if r["ev_por_trade"] > 0 else ""
        print(f"{r['threshold']:>6} {r['trades']:>8} {r['cobertura']:>8.1%} "
              f"{r['winrate']:>8.2%} {r['ev_por_trade']:>+9.4f}{flag}")

    # importancia de features do ultimo fold (referencia)
    imp = pd.Series(last_model.feature_importances_, index=feats)
    print("\nTop 12 features (ultimo fold):")
    print(imp.sort_values(ascending=False).head(12).round(4).to_string())

    Path(args.report).write_text(json.dumps(res, indent=2))
    oos.to_parquet("oos_predictions.parquet", index=False)
    print(f"\nRelatorio salvo em {args.report}; previsoes OOS em oos_predictions.parquet")

    # ---- modelo final para producao ----
    # Replica EXATAMENTE o setup validado no walk-forward: janela de treino
    # de train_days (nao a serie inteira) e numero de arvores fixo = mediana
    # dos folds. Antes, o early stopping usava os ultimos 28 dias e, quando o
    # regime mudava, parava com 2-11 arvores -> modelo quase constante ->
    # Platt com inclinacao ~0 -> p sempre ~0.50 -> NENHUM sinal (out-nov/2026).
    if args.save_model:
        import joblib
        t = pd.to_datetime(ds["candle_open"])
        cut = t.max() - pd.Timedelta(days=args.calib_days)
        tr = (t >= cut - pd.Timedelta(days=args.train_days)) & (t < cut)
        ca = t >= cut
        Xf = ds[feats].astype(float).values
        yf = ds["y"].values
        n_trees = max(30, int(np.median(best_iters)))
        final = make_model(seed=0, n_estimators=n_trees, early_stopping=False)
        final.fit(Xf[tr], yf[tr], verbose=False)
        iso = make_calibrator(args.calib_method)
        iso.fit(final.predict_proba(Xf[ca])[:, 1], yf[ca])

        # ---- trava de seguranca: nao publica modelo "morto" ----
        p_ca = iso.predict(final.predict_proba(Xf[ca])[:, 1])
        thr = args.min_signal_thr
        cobertura = float(((p_ca >= thr) | (p_ca <= 1 - thr)).mean())
        n_sig = int(((p_ca >= thr) | (p_ca <= 1 - thr)).sum())
        pct_call = float((p_ca >= thr).sum() / n_sig) if n_sig else 0.0
        slope = float(iso.lr.coef_[0][0]) if hasattr(iso, "lr") else float("nan")
        info = {"arvores": n_trees, "platt_slope": round(slope, 4),
                "p_min": round(float(p_ca.min()), 4),
                "p_max": round(float(p_ca.max()), 4),
                "cobertura_thr": thr, "cobertura_calib": round(cobertura, 4),
                "pct_call_calib": round(pct_call, 3), "calibracao": args.calib_method}
        res["modelo_final"] = info
        Path(args.report).write_text(json.dumps(res, indent=2))
        print(f"\nModelo final: {info}")

        if not (slope > 0.2 and cobertura >= args.min_coverage):
            raise SystemExit(
                f"[ABORTADO] Modelo final degenerado (slope={slope:.4f}, "
                f"cobertura @{thr}={cobertura:.1%} < {args.min_coverage:.0%}). "
                f"model.joblib NAO foi sobrescrito; o anterior continua valendo.")

        joblib.dump({"model": final, "iso": iso, "features": feats},
                    args.save_model)
        print(f"Modelo final salvo em {args.save_model}")

    print("\nCRITERIO DE DECISAO honesto:")
    print("  - AUC ~0.50 e nenhum threshold com EV>0 estavel entre folds")
    print("    => o sinal nao existe; nao coloque dinheiro.")
    print("  - EV>0 apenas com cobertura minuscula (<2%) => provavel ruido.")
    print("  - So considere ir adiante se o EV>0 se repetir na MAIORIA dos folds.")


if __name__ == "__main__":
    main()

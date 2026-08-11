"""
calibration.py — calibradores de probabilidade (importavel por qualquer script).
Precisa estar na mesma pasta de quem carrega o model.joblib.
"""

import numpy as np


class PlattCalibrator:
    """Calibracao sigmoide (2 parametros). Bem mais estavel que isotonic
    em janelas pequenas — isotonic vira funcao-degrau overfittada."""

    def fit(self, p_raw, y):
        from sklearn.linear_model import LogisticRegression
        z = self._logit(p_raw)
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")
        self.lr.fit(z.reshape(-1, 1), y)
        return self

    def predict(self, p_raw):
        z = self._logit(np.asarray(p_raw, dtype=float))
        return self.lr.predict_proba(z.reshape(-1, 1))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

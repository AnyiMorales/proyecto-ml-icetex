"""Métricas de regresión implementadas a mano."""

import numpy as np


def mse(y, y_pred):
    """Riesgo empírico con pérdida cuadrática: (1/n) * sum (y_i - h(x_i))^2."""
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y - y_pred) ** 2))


def rmse(y, y_pred):
    """Raíz del MSE. Queda en las mismas unidades que y (aquí, años)."""
    return float(np.sqrt(mse(y, y_pred)))


def mae(y, y_pred):
    """Error absoluto medio: (1/n) * sum |y_i - h(x_i)|."""
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y - y_pred)))


def r2(y, y_pred):
    """
    Coeficiente de determinación: 1 - SS_res / SS_tot.
    1 es un ajuste perfecto y 0 equivale a predecir siempre la media de y.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def resumen(y, y_pred):
    """Diccionario con MSE, RMSE, MAE y R2."""
    return {"MSE": mse(y, y_pred), "RMSE": rmse(y, y_pred), "MAE": mae(y, y_pred), "R2": r2(y, y_pred)}


def diferencia_bootstrap(y, pred_a, pred_b, repeticiones=1000, semilla=0):
    """
    Diferencia de RMSE entre dos modelos, RMSE(a) - RMSE(b), con intervalo de
    confianza del 95 % por bootstrap pareado: se remuestrean las MISMAS personas
    para ambos modelos, así la comparación no depende de qué personas tocaron.

    Si el intervalo contiene el 0, los modelos no se distinguen con estos datos.
    Devuelve (diferencia promedio, límite inferior, límite superior).
    """
    rng = np.random.default_rng(semilla)
    y = np.asarray(y, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y)
    diferencias = np.empty(repeticiones)
    for b in range(repeticiones):
        idx = rng.integers(0, n, n)
        diferencias[b] = rmse(y[idx], pred_a[idx]) - rmse(y[idx], pred_b[idx])
    inferior, superior = np.percentile(diferencias, [2.5, 97.5])
    return float(diferencias.mean()), float(inferior), float(superior)

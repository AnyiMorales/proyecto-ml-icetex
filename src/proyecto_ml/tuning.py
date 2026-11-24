"""Curvas de complejidad y búsqueda de hiperparámetros."""

import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

from .data import SEMILLA
from .metrics import rmse
from .models import construir_modelo

PARTICIONES = 5
SCORING = "neg_root_mean_squared_error"   # scikit-learn maximiza, por eso el RMSE va en negativo


def curva_complejidad(nombre, parametro, valores, X_tr, y_tr, X_val, y_val, **fijos):
    """
    Para cada valor del hiperparámetro `parametro`, ajusta el modelo con (X_tr, y_tr)
    y mide el RMSE en entrenamiento y en validación.

    Devuelve un DataFrame con columnas: <parametro>, rmse_train, rmse_val.
    """
    filas = []
    for valor in valores:
        modelo = construir_modelo(nombre, **fijos, **{parametro: valor})
        modelo.fit(X_tr, y_tr)
        filas.append(
            {
                parametro: valor,
                "rmse_train": rmse(y_tr, modelo.predict(X_tr)),
                "rmse_val": rmse(y_val, modelo.predict(X_val)),
            }
        )
    return pd.DataFrame(filas)


def _validacion_cruzada():
    return KFold(n_splits=PARTICIONES, shuffle=True, random_state=SEMILLA)


def busqueda_grid(nombre, grilla, X, y, **fijos):
    """
    Grid search: prueba TODAS las combinaciones de `grilla` con validación cruzada.
    `grilla` es un diccionario {hiperparámetro: lista de valores}.
    Devuelve (búsqueda ajustada, segundos que tardó).
    """
    parametros = {f"modelo__{k}": v for k, v in grilla.items()}
    busqueda = GridSearchCV(
        construir_modelo(nombre, **fijos),
        parametros,
        scoring=SCORING,
        cv=_validacion_cruzada(),
    )
    inicio = time.time()
    busqueda.fit(X, y)
    return busqueda, time.time() - inicio


def busqueda_random(nombre, distribuciones, X, y, n_iter, **fijos):
    """
    Random search: prueba `n_iter` combinaciones sorteadas de `distribuciones`
    (listas de valores o distribuciones de scipy) con validación cruzada.
    Devuelve (búsqueda ajustada, segundos que tardó).
    """
    parametros = {f"modelo__{k}": v for k, v in distribuciones.items()}
    busqueda = RandomizedSearchCV(
        construir_modelo(nombre, **fijos),
        parametros,
        n_iter=n_iter,
        scoring=SCORING,
        cv=_validacion_cruzada(),
        random_state=SEMILLA,
    )
    inicio = time.time()
    busqueda.fit(X, y)
    return busqueda, time.time() - inicio


def resultados_cv(busqueda, k=10):
    """
    Las k mejores combinaciones: hiperparámetros, RMSE promedio en validación
    cruzada y su desviación.
    """
    r = pd.DataFrame(busqueda.cv_results_)
    tabla = pd.DataFrame(
        {
            "hiperparámetros": [
                {p.replace("modelo__", ""): v for p, v in fila.items()} for fila in r["params"]
            ],
            "rmse_cv": -r["mean_test_score"],
            "desv_cv": r["std_test_score"],
            "ranking": r["rank_test_score"],
        }
    )
    return tabla.sort_values("ranking").head(k).reset_index(drop=True)


def mejores_hiperparametros(busqueda):
    """Mejores hiperparámetros sin el prefijo 'modelo__'."""
    return {p.replace("modelo__", ""): v for p, v in busqueda.best_params_.items()}

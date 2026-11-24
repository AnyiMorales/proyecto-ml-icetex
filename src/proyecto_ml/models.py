"""Pipelines de preprocesamiento + modelo para los cinco candidatos."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data import FEATURES_CATEGORICAS, FEATURES_NUMERICAS, SEMILLA

LINEA_BASE = "Línea base (media)"
LINEAL = "Regresión lineal"
ARBOL = "Árbol de decisión"
BOSQUE = "Random Forest"
KNN = "kNN"
NOMBRES = [LINEA_BASE, LINEAL, ARBOL, BOSQUE, KNN]


def preprocesador():
    """
    Variables numéricas: se estandarizan (media 0, desviación 1). Esto importa para
    kNN, que usa distancias, y no afecta a los árboles.
    Variables categóricas: one-hot (una columna 0/1 por categoría); las categorías
    que no aparecen en entrenamiento se ignoran.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), FEATURES_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAS),
        ]
    )


def _estimador(nombre):
    estimadores = {
        LINEA_BASE: DummyRegressor(strategy="mean"),
        LINEAL: LinearRegression(),
        ARBOL: DecisionTreeRegressor(random_state=SEMILLA),
        BOSQUE: RandomForestRegressor(random_state=SEMILLA),
        KNN: KNeighborsRegressor(),
    }
    return estimadores[nombre]


def construir_modelo(nombre, **hiperparametros):
    """
    Devuelve un Pipeline (preprocesador -> estimador) sin ajustar.

    Los hiperparámetros se pasan con su nombre normal, por ejemplo
    construir_modelo(ARBOL, max_depth=5), o con el prefijo "modelo__" que
    entregan las búsquedas de scikit-learn; ambas formas funcionan.
    """
    pipeline = Pipeline([("prep", preprocesador()), ("modelo", _estimador(nombre))])
    parametros = {
        (k if k.startswith("modelo__") else f"modelo__{k}"): v for k, v in hiperparametros.items()
    }
    return pipeline.set_params(**parametros)

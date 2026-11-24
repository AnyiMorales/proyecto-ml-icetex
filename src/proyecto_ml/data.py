"""Carga y limpieza de la base de convocatorias de ICETEX (Talento TI)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .simulation import agregar_objetivo

# .../proyecto_final/src/proyecto_ml/data.py -> sube 2 niveles -> .../proyecto_final
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ARCHIVO = (
    "Convocatorias_otorgadas_para_competencias_gerenciales_y_transversales"
    "_-_Talento_TI_20240827.csv"
)


def cargar_convocatorias(archivo=ARCHIVO):
    """
    Carga la base y devuelve un DataFrame limpio con estas columnas:

        estrato, anio_inicio, valor_comprometido, genero, departamento,
        ciudad, programa, tipo_institucion, programa_academico

    Limpieza (la misma del informe anterior, más el formato del valor):
      - se eliminan los 19 registros sin estrato;
      - VALOR COMPROMETIDO viene como texto con comas ("4,301,536.8") y se
        convierte a número;
      - de CONVOCATORIA se toma solo el año;
      - GENERO trae etiquetas mezcladas (M, F, Masculino, Femenino) y se unifica;
      - los 12 registros sin ciudad se marcan como "SIN DATO".
    """
    df = pd.read_csv(DATA_DIR / archivo)
    df = df.dropna(subset=["ESTRATO"]).copy()

    limpio = pd.DataFrame(index=df.index)
    limpio["estrato"] = pd.to_numeric(df["ESTRATO"]).astype(int)
    limpio["anio_inicio"] = pd.to_datetime(
        df["CONVOCATORIA"], format="%m/%d/%Y %I:%M:%S %p"
    ).dt.year
    limpio["valor_comprometido"] = pd.to_numeric(
        df["VALOR COMPROMETIDO"].astype(str).str.replace(",", "", regex=False)
    )
    limpio["genero"] = (
        df["GENERO"].astype(str).str.strip().str.upper().str[0].map({"M": "M", "F": "F"})
    )
    limpio["departamento"] = df["DEPARTAMENTO DE RESIDENCIA"].fillna("SIN DATO")
    limpio["ciudad"] = df["CIUDAD DE RESIDENCIA"].fillna("SIN DATO")
    limpio["programa"] = df["PROGRAMA"]
    limpio["tipo_institucion"] = df["TIPO INSTITUCION"]
    limpio["programa_academico"] = df["PROGRAMA ACADEMICO"]
    return limpio.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Variables del modelado
# ---------------------------------------------------------------------------
# El simulador solo usa estrato, anio_inicio, valor_comprometido y programa.
# genero, tipo_institucion y departamento quedan como variables sin efecto real,
# para ver si los modelos las ignoran. Se descartan ciudad (muchísimas categorías)
# y programa_academico (48 categorías) para mantener el problema manejable.
FEATURES_NUMERICAS = ["estrato", "anio_inicio", "valor_comprometido"]
FEATURES_CATEGORICAS = ["programa", "tipo_institucion", "genero", "departamento"]
OBJETIVO = "anios_pago"
SEMILLA = 2026


def cargar_dataset(semilla=SEMILLA):
    """Base limpia con la variable objetivo simulada `anios_pago`."""
    return agregar_objetivo(cargar_convocatorias(), semilla=semilla)


def particionar(datos, semilla=SEMILLA, prueba=0.2):
    """
    Separa 80 % para desarrollo (entrenamiento y selección de modelos) y 20 % para
    prueba. La prueba solo se usa al final, en la comparación de modelos.

    Devuelve X_dev, X_test, y_dev, y_test. Los índices originales se conservan.
    """
    X = datos[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y = datos[OBJETIVO]
    return train_test_split(X, y, test_size=prueba, random_state=semilla)

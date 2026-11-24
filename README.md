# Trabajo final — Introducción al Machine Learning

Universidad El Bosque, Programa de Matemáticas y Estadística.

**Integrante:** Anyi Morales

## Objetivo

Comparar modelos de regresión (línea base, regresión lineal, árbol de decisión, Random Forest y kNN),
ajustando sus hiperparámetros, para predecir los **años que una persona tarda en pagar su crédito de ICETEX**.

**Aviso importante:** los años de pago no están en la base de ICETEX, se **simulan** con un modelo de supuestos
declarados (`src/proyecto_ml/simulation.py` y `notebooks/01_simulacion.ipynb`). Las conclusiones sobre los modelos
aplican a ese simulador y no al comportamiento real de ICETEX.

## Estructura

- `data/` — base "Convocatorias otorgadas para competencias gerenciales y transversales - Talento TI" (descargada el 2024-08-27).
- `src/proyecto_ml/` — código reutilizable:
  - `data.py`: carga, limpieza y partición (80 % desarrollo, 20 % prueba).
  - `simulation.py`: simulador de los años de pago y piso de error del azar.
  - `metrics.py`: MSE, RMSE, MAE, R² y bootstrap pareado, implementados a mano.
  - `models.py`: pipelines de preprocesamiento + modelo.
  - `tuning.py`: curvas de complejidad, grid search y random search.
- `notebooks/` — cuadernos ejecutados, para correr **en este orden**:
  1. `01_simulacion.ipynb`: variable objetivo, supuestos y comprobación de que no es una fórmula exacta.
  2. `02_particion_y_modelos_base.ipynb`: pipeline, línea base y modelos con hiperparámetros por defecto.
  3. `03_complejidad_e_hiperparametros.ipynb`: curvas de complejidad (error de entrenamiento vs validación).
  4. `04_busqueda_de_hiperparametros.ipynb`: grid search y random search con validación cruzada.
  5. `05_comparacion_final.ipynb`: comparación en prueba, piso de error, bootstrap e importancia de variables.
- `resultados/` — figuras y `mejores_hiperparametros.json` (lo escribe el cuaderno 04 y lo lee el 05).

## Decisiones de diseño

- La prueba (20 %) solo se usa en el cuaderno 05. La selección de modelos se hace con validación cruzada de 5 particiones.
- Variables predictoras: estrato, año de inicio, valor comprometido, programa, tipo de institución, género y departamento.
  El simulador solo usa las cuatro primeras y programa; género, tipo de institución y departamento no tienen efecto, a propósito.
- El Random Forest usa 50 árboles en las búsquedas para que tarden menos.

## Instalación

```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Ejecución

Abre los cuadernos de `notebooks/` en orden, selecciona el kernel del entorno `venv` y ejecuta todas las celdas.
Con un solo núcleo de CPU, el cuaderno 04 (el más pesado) tarda un par de minutos.

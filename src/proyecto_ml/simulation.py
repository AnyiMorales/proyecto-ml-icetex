"""
Simulador de los años que tarda una persona en pagar su crédito.

IMPORTANTE: los años de pago NO están en la base de ICETEX. Se simulan a partir
de supuestos, y todos están declarados aquí abajo. Los resultados de los modelos
describen este simulador, no el comportamiento real de ICETEX.

Idea del modelo: cada año la persona abona un porcentaje de lo que gana, que
depende de su estrato. Los años de pago son el tiempo que tarda el abono
acumulado en alcanzar el valor comprometido.
"""

import numpy as np

# --- Supuesto 1: salario mínimo mensual por año (con guion bajo como separador de
# miles; escribir 616.000 en Python sería 616.0, el error del informe anterior).
SMMLV = {
    2014: 616_000,
    2015: 644_350,
    2016: 689_455,
    2017: 737_717,
    2018: 781_242,
    2019: 828_116,
    2020: 877_803,
    2021: 908_526,
    2022: 1_000_000,
    2023: 1_160_000,
    2024: 1_300_000,
}

# --- Supuesto 2: porcentaje del ingreso anual que se abona, según el estrato.
PORCENTAJE_POR_ESTRATO = {e: 0.025 + 0.02 * (e - 1) for e in range(1, 7)}  # 2,5 % ... 12,5 %

# --- Supuesto 3: variabilidad individual y efecto del programa.
MESES_MIN, MESES_MAX = 8, 12      # meses trabajados por año, al azar (media 10, como en el informe)
SIGMA_INGRESO = 0.25              # dispersión del factor de ingreso de cada persona (lognormal)
FACTOR_GERENCIALES = 1.25         # quienes hacen Competencias Gerenciales ganan 25 % más (supuesto)

HORIZONTE = 60                    # años máximos que se simulan
PROGRAMA_GERENCIAL = "COMPETENCIAS GERENCIALES"


def crecimiento_promedio_smmlv():
    """Crecimiento anual compuesto del salario mínimo entre 2014 y 2024 (≈ 7,7 %)."""
    return (SMMLV[2024] / SMMLV[2014]) ** (1 / 10) - 1


def smmlv_en(anios, crecimiento):
    """
    Salario mínimo mensual para cada año del arreglo `anios`.
    Hasta 2024 usa la tabla; después de 2024 crece a la tasa `crecimiento`.
    """
    anios = np.asarray(anios)
    tabla_anios = np.array(sorted(SMMLV))
    tabla_valores = np.array([SMMLV[a] for a in tabla_anios], dtype=float)
    dentro = np.interp(anios, tabla_anios, tabla_valores)
    fuera = tabla_valores[-1] * (1 + crecimiento) ** (anios - tabla_anios[-1])
    return np.where(anios <= tabla_anios[-1], dentro, fuera)


def simular_anios_de_pago(df, semilla=2026, con_azar=True):
    """
    Simula los años de pago de cada persona de `df`.

    Cada año k (k = 0, 1, 2, ...) la persona abona:
        pago_k = porcentaje(estrato) * SMMLV(año_inicio + k) * meses_k * factor_persona * factor_programa
    Los años de pago son el primer k en que el abono acumulado alcanza el valor
    comprometido, más la fracción del último año que hace falta (interpolación
    lineal), así que el resultado es continuo (por ejemplo 7,4 años).

    Parámetros
    ----------
    df       : DataFrame con estrato, anio_inicio, valor_comprometido y programa
    semilla  : semilla del generador aleatorio (resultados reproducibles)
    con_azar : si es False, se quita el azar (meses = 10, factor de persona = 1);
               sirve para ver qué parte del resultado es fórmula pura.

    Devuelve (anios_pago, censurados): los años simulados y una máscara de las
    personas que no terminaron de pagar dentro del horizonte (su valor es HORIZONTE).
    """
    rng = np.random.default_rng(semilla)
    n = len(df)
    valor = df["valor_comprometido"].to_numpy(dtype=float)
    anio0 = df["anio_inicio"].to_numpy()
    porcentaje = df["estrato"].map(PORCENTAJE_POR_ESTRATO).to_numpy()
    factor_programa = np.where(df["programa"].to_numpy() == PROGRAMA_GERENCIAL, FACTOR_GERENCIALES, 1.0)

    if con_azar:
        factor_persona = rng.lognormal(mean=0.0, sigma=SIGMA_INGRESO, size=n)  # fijo para cada persona
        meses = rng.integers(MESES_MIN, MESES_MAX + 1, size=(n, HORIZONTE))   # cambia cada año
    else:
        factor_persona = np.ones(n)
        meses = np.full((n, HORIZONTE), 10)

    anios = anio0[:, None] + np.arange(HORIZONTE)[None, :]                     # (n, HORIZONTE)
    ingreso_mensual = smmlv_en(anios, crecimiento_promedio_smmlv())
    pago = porcentaje[:, None] * ingreso_mensual * meses * (factor_persona * factor_programa)[:, None]
    acumulado = np.cumsum(pago, axis=1)

    alcanzo = acumulado >= valor[:, None]
    censurados = ~alcanzo.any(axis=1)
    k = alcanzo.argmax(axis=1)                       # primer año en que se alcanza el valor
    filas = np.arange(n)
    acumulado_previo = np.where(k > 0, acumulado[filas, k - 1], 0.0)
    fraccion = (valor - acumulado_previo) / pago[filas, k]

    anios_pago = k + fraccion
    anios_pago[censurados] = HORIZONTE
    return anios_pago, censurados


def agregar_objetivo(df, semilla=2026, con_azar=True):
    """Devuelve una copia de `df` con la columna `anios_pago` (la variable objetivo)."""
    anios_pago, _ = simular_anios_de_pago(df, semilla=semilla, con_azar=con_azar)
    resultado = df.copy()
    resultado["anios_pago"] = anios_pago
    return resultado


def piso_de_error(df, repeticiones=100, semilla_base=10_000):
    """
    Error imposible de evitar por el azar del simulador.

    Simula `repeticiones` veces la variable objetivo de cada persona (con semillas
    distintas a la del objetivo real) y calcula, para cada una, el promedio y la
    varianza. El promedio es el mejor predictor posible (conoce estrato, año,
    valor y programa, pero no el azar); la varianza es su error irreducible.

    Devuelve (promedio, varianza), dos arreglos con una entrada por persona.
    """
    suma = np.zeros(len(df))
    suma_cuadrados = np.zeros(len(df))
    for r in range(repeticiones):
        anios, _ = simular_anios_de_pago(df, semilla=semilla_base + r)
        suma += anios
        suma_cuadrados += anios**2
    promedio = suma / repeticiones
    varianza = (suma_cuadrados / repeticiones - promedio**2) * repeticiones / (repeticiones - 1)
    return promedio, varianza

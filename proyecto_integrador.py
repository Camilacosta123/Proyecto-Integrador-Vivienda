# =====================================================================
# PROYECTO INTEGRADOR - Política monetaria, costo del crédito y actividad edificadora de vivienda
# Un solo script: carga la base una vez y corre las 5 partes en orden.
#   Parte 1: series y estacionariedad (ADF)      Parte 4: cadena de Markov
#   Parte 2: selección de rezagos y VAR(3)       Parte 5: simulación de Monte Carlo
#   Parte 3: diagnósticos, impulso-respuesta, pronóstico
# Base: base_integrador.xlsx, hoja "Base_Analizada" (BanRep: TPM y tasa de colocación; DANE: área aprobada).
# =====================================================================
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

# ---------- CARGA DE DATOS (una sola vez) ----------
import os
_AQUI = os.path.dirname(os.path.abspath(__file__))
# Busca la base en data/ y, si no está, suelta junto a este archivo (funciona con ambas estructuras del repositorio)
ruta = next((p for p in (os.path.join(_AQUI, "data", "base_integrador.xlsx"), os.path.join(_AQUI, "base_integrador.xlsx"))
             if os.path.exists(p)), "base_integrador.xlsx")
datos = pd.read_excel(ruta, sheet_name="Base_Analizada")
datos["fecha"] = pd.to_datetime(datos["fecha"])
df = datos.set_index("fecha"); df.index = df.index.to_period("M")
df["ln_area"] = np.log(df["area_vivienda_77mun_m2"])      # logaritmo: serie positiva y de escala amplia
base = df[["tpm", "tasa_colocacion_total", "ln_area"]].copy()
print("Observaciones:", len(base), "| Periodo:", base.index.min(), "a", base.index.max())
print("Valores faltantes:", int(base.isna().sum().sum()))


# =====================  PARTE 1: SERIES Y ESTACIONARIEDAD  =====================
# 2) Gráficos de las series en nivel ------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for a, c, t in zip(ax, base.columns,
                   ["TPM (%)", "Tasa de colocación total (%)", "ln(área aprobada vivienda, m²)"]):
    a.plot(base.index.to_timestamp(), base[c]); a.set_title(t); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_series_nivel.png", dpi=130); plt.close()

# 3) Estacionalidad del área aprobada (promedio de ln_area por mes) ------
est = base["ln_area"].groupby(base.index.month).mean().round(3)
print("\nPromedio de ln_area por mes del año:\n", est.to_string())

# 4) Pruebas ADF: nivel y primera diferencia ----------------------------
def adf(s, nombre):
    r = adfuller(s.dropna(), regression="c", autolag="AIC")
    return {"serie": nombre, "estadístico": round(r[0], 3), "p-valor": round(r[1], 3),
            "rezagos usados": r[2], "n": r[3]}
filas = []
for c in base.columns:
    filas.append(adf(base[c], c + " (nivel)"))
    filas.append(adf(base[c].diff(), c + " (1ª diferencia)"))
print("\nPrueba ADF (H0: raíz unitaria; constante incluida):")
print(pd.DataFrame(filas).to_string(index=False))


# =====================  PARTE 2: SELECCIÓN DE REZAGOS Y VAR  =====================
# 2.1) Transformaciones (DECISIÓN: tasas en primera diferencia; ln_area en nivel) ---
modelo = pd.DataFrame({
    "d_tpm":   df["tpm"].diff(),                      # cambio mensual de la TPM (pp)
    "d_tasa":  df["tasa_colocacion_total"].diff(),    # cambio mensual de la tasa de colocación (pp)
    "ln_area": df["ln_area"],                         # nivel (estacionaria según ADF)
}).dropna()
dic = pd.DataFrame({"dic": (modelo.index.month == 12).astype(float)}, index=modelo.index)  # dummy diciembre
modelo.index = modelo.index.to_timestamp(); modelo = modelo.asfreq("MS"); dic.index = modelo.index
print("Muestra del VAR:", modelo.index.min().strftime("%Y-%m"), "a", modelo.index.max().strftime("%Y-%m"), "| n =", len(modelo))

# 2.2) Selección de rezagos (máx. 12) -------------------------------------
var = VAR(modelo, exog=dic)
sel = var.select_order(maxlags=12)
print("\nCriterios de información por rezagos:")
print(sel.summary())
print("Rezagos sugeridos:", sel.selected_orders)

# 2.3) Estimación con el rezago elegido (cámbialo según tu decisión) ------
LAGS = 3
res = var.fit(LAGS)
print(f"\n===== VAR({LAGS}) =====")
print(res.summary())

# 2.4) Estabilidad: todas las raíces del polinomio deben estar fuera del círculo unitario
print("\nMódulo de las raíces (deben ser > 1):", np.round(np.abs(res.roots), 3))
print("¿Estable?:", res.is_stable())

# 2.5) Causalidad de Granger (sentido predictivo, no estructural) ---------
print("\nPruebas de Granger (H0: la variable NO ayuda a predecir al bloque):")
vars_ = list(modelo.columns)
for causa in vars_:
    for efecto in vars_:
        if causa != efecto:
            t = res.test_causality(efecto, [causa], kind="f")
            print(f"{causa:8s} -> {efecto:8s}  F={t.test_statistic:7.3f}  p={t.pvalue:.4f}")


# =====================  PARTE 3: DIAGNÓSTICOS, IMPULSO-RESPUESTA Y PRONÓSTICO  =====================
# 4) Diagnósticos de residuos --------------------------------------------
print("\n--- Autocorrelación residual (Portmanteau, H0: ruido blanco), 12 rezagos ---")
w = res.test_whiteness(nlags=12, adjusted=True)
print("estadístico = %.2f | gl = %d | p = %.4f" % (w.test_statistic, w.df, w.pvalue))
print("\n--- Normalidad de residuos (Jarque-Bera conjunto, H0: normal) ---")
nt = res.test_normality()
print("estadístico = %.2f | gl = %d | p = %.4f" % (nt.test_statistic, nt.df, nt.pvalue))
print("\nDesviación estándar de residuos:\n", res.resid.std().round(3).to_string())

# 5) Funciones de impulso-respuesta (ordenamiento de Cholesky: d_tpm, d_tasa, ln_area)
# El orden supone: la TPM no responde contemporáneamente a las otras; el área responde a las tasas.
irf = res.irf(12)
fig = irf.plot(orth=True, impulse="d_tpm", signif=0.05)
fig.suptitle("Respuesta a un choque de 1 desv. est. en d_tpm (Cholesky, IC 95%)")
fig.savefig("fig_irf_tpm.png", dpi=130)
fig2 = irf.plot(orth=True, impulse="d_tasa", signif=0.05)
fig2.suptitle("Respuesta a un choque de 1 desv. est. en d_tasa (Cholesky, IC 95%)")
fig2.savefig("fig_irf_tasa.png", dpi=130)
print("\nRespuesta de ln_area (en logs) a un choque de 1 desv. est. en d_tpm, meses 1 a 12:")
print(np.round(irf.orth_irfs[1:, 2, 0], 4))

# 6) Descomposición de varianza del error de pronóstico (mes 12) --------
fev = res.fevd(12)
print("\nDescomposición de varianza a 12 meses (filas = variable explicada):")
print(pd.DataFrame(fev.decomp[:, -1, :], index=modelo.columns, columns=modelo.columns).round(3))

# 7) Pronóstico a 12 meses (octubre 2026 a septiembre 2027) --------------
HZ = 12
fechas = pd.date_range(modelo.index[-1] + pd.offsets.MonthBegin(1), periods=HZ, freq="MS")
exog_fut = (fechas.month == 12).astype(float).reshape(-1, 1)
fc_var = res.forecast(modelo.values[-LAGS:], steps=HZ, exog_future=exog_fut)
fc_var = pd.DataFrame(fc_var, index=fechas, columns=modelo.columns)
tpm_ult = df["tpm"].iloc[-1]; tasa_ult = df["tasa_colocacion_total"].iloc[-1]
pron = pd.DataFrame({
    "tpm": tpm_ult + fc_var["d_tpm"].cumsum(),
    "tasa_colocacion_total": tasa_ult + fc_var["d_tasa"].cumsum(),
    "area_vivienda_m2": np.exp(fc_var["ln_area"]),
})
print("\nPronóstico (niveles reconstruidos a partir de las diferencias):")
print(pron.round(2).to_string())

# 8) Evaluación fuera de muestra: se estima hasta 2025-09 y se pronostican 12 meses --
corte = modelo.index[-1] - pd.DateOffset(months=12)
ent, prueba = modelo.loc[:corte], modelo.loc[corte + pd.offsets.MonthBegin(1):]
r2 = VAR(ent, exog=dic.loc[ent.index]).fit(LAGS)
f2 = r2.forecast(ent.values[-LAGS:], steps=len(prueba), exog_future=dic.loc[prueba.index].values)
f2 = pd.DataFrame(f2, index=prueba.index, columns=modelo.columns)
ing = lambda e: np.sqrt(((e) ** 2).mean())
print("\nRMSE fuera de muestra (12 meses) VAR vs ingenuo (último valor / cambio 0):")
for c in modelo.columns:
    ref = ent[c].iloc[-1] if c == "ln_area" else 0.0
    print("%-8s VAR = %.3f | ingenuo = %.3f" % (c, ing(prueba[c] - f2[c]), ing(prueba[c] - ref)))


# =====================  PARTE 4: CADENA DE MARKOV  =====================
ln_area_s = df["ln_area"]
# 2) Desestacionalizar restando el promedio de cada mes (quita el pico de diciembre) ---
factor = ln_area_s.groupby(ln_area_s.index.month).transform("mean") - ln_area_s.mean()
ln_adj = ln_area_s - factor

# 3) Estados por terciles de la serie desestacionalizada (criterio reproducible) ---
q1, q2 = ln_adj.quantile([1/3, 2/3])
est = pd.cut(ln_adj, bins=[-np.inf, q1, q2, np.inf], labels=["Bajo", "Medio", "Alto"])
print("Umbrales (m², área desestacionalizada): Bajo <= %.0f | Medio <= %.0f | Alto > %.0f" % (np.exp(q1), np.exp(q2), np.exp(q2)))
print("\nMeses en cada estado:\n", est.value_counts().reindex(["Bajo", "Medio", "Alto"]).to_string())
print("Estado actual (", est.index[-1], "):", est.iloc[-1])

# 4) Conteo de transiciones y matriz (filas = estado actual, columnas = estado siguiente) ---
orden = ["Bajo", "Medio", "Alto"]
origen, destino = est.iloc[:-1].values, est.iloc[1:].values
conteo = pd.crosstab(pd.Series(origen, name="actual"), pd.Series(destino, name="siguiente")).reindex(index=orden, columns=orden, fill_value=0)
print("\nConteo de transiciones:\n", conteo)
P_mk = conteo.div(conteo.sum(axis=1), axis=0)
print("\nMatriz de transición P:\n", P_mk.round(3))
print("Suma de cada fila (debe ser 1):", P_mk.sum(axis=1).round(6).values)

# 5) Interpretación cuantitativa ----------------------------------------
dur = 1 / (1 - np.diag(P_mk.values))
print("\nDuración esperada de cada estado (meses):", dict(zip(orden, [round(float(x),2) for x in dur])))
vals, vecs = np.linalg.eig(P_mk.values.T)
pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); pi = pi / pi.sum()
print("Distribución estacionaria:", dict(zip(orden, [round(float(x),3) for x in pi])))
p0 = np.zeros(3); p0[orden.index(str(est.iloc[-1]))] = 1
for k in (3, 6, 12):
    print("Probabilidades a %2d meses desde el estado actual:" % k, dict(zip(orden, [round(float(x),3) for x in (p0 @ np.linalg.matrix_power(P_mk.values, k))])))

# 6) Chequeo de homogeneidad temporal: primera mitad vs segunda mitad de la muestra ---
m = len(origen) // 2
for nombre, o, d in (("2005-2015", origen[:m], destino[:m]), ("2015-2026", origen[m:], destino[m:])):
    cm = pd.crosstab(pd.Series(o), pd.Series(d)).reindex(index=orden, columns=orden, fill_value=0)
    print("\nMatriz en", nombre, ":\n", cm.div(cm.sum(axis=1), axis=0).round(2))

# 7) ¿Hay una tendencia? Estados por año (los terciles se calcularon con toda la muestra) ---
print("\nDistribución de estados por periodo (número de meses):")
print(pd.crosstab(pd.Series(est.index.year // 5 * 5, index=est.index, name="lustro"), est))

# =====================  PARTE 5: SIMULACIÓN DE MONTE CARLO  =====================
# Se reutilizan el VAR(3) ya estimado (res) y los datos del modelo de la Parte 2.
K = modelo.shape[1]
A = res.coefs                                  # (LAGS, K, K): efecto de cada rezago
c = res.params.loc["const"].values             # constantes
bd = res.params.loc["dic"].values              # coeficiente de la dummy de diciembre
E = res.resid.values                           # residuos (n x K) para el bootstrap

# 5.2) Parámetros del experimento (SUPUESTOS EXPLÍCITOS) -------------------
H = 12                  # horizonte: 12 meses (oct-2026 a sep-2027)
N = 10000               # número de trayectorias
SEMILLA = 2026          # semilla fija para que el resultado se repita
fechas = pd.date_range(modelo.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
D = (fechas.month == 12).astype(float)
ESCENARIOS = {"Favorable (choque -0,10 pp/mes x 4)": -0.10, "Central (sin choque impuesto)": 0.0,
              "Adverso (choque +0,10 pp/mes x 4)": +0.10}

# Nota: el choque es una innovación sobre d_tpm; el VAR lo amplifica por la inercia de d_tpm.
#       Por eso se reporta la mediana de la TPM al mes 12 en cada escenario.

# 5.3) Umbrales de referencia (sale de los datos) --------------------------
area = df["area_vivienda_77mun_m2"]
prom12 = area.rolling(12).mean().dropna()
UMBRAL = prom12.quantile(0.25)                 # "crítico" = percentil 25 del promedio móvil de 12 meses
# (factor estacional y Q1 ya definidos en la Parte 4; se recalculan aquí para que esta parte sea legible por sí sola)
ln_area_s = df["ln_area"]
factor = ln_area_s.groupby(ln_area_s.index.month).transform("mean") - ln_area_s.mean()
ln_adj = ln_area_s - factor
Q1 = ln_adj.quantile(1/3)                      # umbral del estado "Bajo" (igual que el Bloque 4)
fac_mes12 = factor.groupby(factor.index.month).mean().loc[fechas[-1].month]
print("Umbral crítico (percentil 25 del promedio móvil de 12 meses): %.0f m²/mes" % UMBRAL)
print("Promedio de los últimos 12 meses observados: %.0f m²/mes" % area.iloc[-12:].mean())

# 5.4) Simulador -----------------------------------------------------------
def simular(shock_tpm, n=N, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    hist = modelo.values[-LAGS:].copy()
    y = np.repeat(hist[None, :, :], n, axis=0)                  # (n, LAGS, K)
    trayect = np.zeros((n, H, K))
    for t in range(H):
        e = E[rng.integers(0, len(E), size=n)]                  # se remuestrean filas completas de residuos
        mu = c + D[t] * bd
        for l in range(LAGS):
            mu = mu + y[:, -1 - l, :] @ A[l].T
        nuevo = mu + e
        if t < 4:
            nuevo[:, 0] += shock_tpm                            # choque impuesto sobre d_tpm en los primeros 4 meses
        trayect[:, t, :] = nuevo
        y = np.concatenate([y[:, 1:, :], nuevo[:, None, :]], axis=1)
    return trayect

# 5.5) Verificación: sin ruido debe coincidir con el pronóstico del VAR ------
E_cero = E.copy(); E = np.zeros_like(E)
det = simular(0.0, n=1)[0]
E = E_cero
fc_chk = res.forecast(modelo.values[-LAGS:], steps=H, exog_future=D.reshape(-1, 1))
print("Chequeo (trayectoria sin ruido vs res.forecast): diferencia máx. = %.2e" % np.abs(det - fc_chk).max())

# 5.6) Resultados por escenario ----------------------------------------------
tpm0 = df["tpm"].iloc[-1]; Pbajo_markov = float((p0 @ np.linalg.matrix_power(P_mk.values, 12))[0])
filas = []
for nombre, s in ESCENARIOS.items():
    tr = simular(s)
    area_m = np.exp(tr[:, :, 2])                                # m² mensuales simulados
    prom = area_m.mean(axis=1)                                  # promedio de 12 meses por trayectoria
    tpm12 = tpm0 + tr[:, :, 0].sum(axis=1)                      # nivel de TPM al mes 12
    estado_bajo = (tr[:, -1, 2] - fac_mes12) <= Q1              # ¿estado "Bajo" en el mes 12?
    p5, p50, p95 = np.percentile(prom, [5, 50, 95])
    filas.append({"escenario": nombre, "área prom. P5": p5, "P50": p50, "P95": p95,
                  "P(prom < umbral)": (prom < UMBRAL).mean(), "P(estado Bajo mes 12)": estado_bajo.mean(),
                  "TPM mes 12 (mediana)": np.median(tpm12)})
print("\nResultados (N = %d trayectorias, semilla = %d):" % (N, SEMILLA))
out = pd.DataFrame(filas).set_index("escenario")
pd.options.display.float_format = "{:,.3f}".format
print(out.round(3).to_string())
print("\nReferencia de Markov: P(Bajo a 12 meses desde el estado actual) = %.3f" % Pbajo_markov)

# 5.7) Estabilidad numérica: misma probabilidad con otras semillas y tamaños ---
print("\nEstabilidad de P(prom < umbral) en el escenario central:")
for n, sd in [(2000, 1), (5000, 2), (10000, 3), (10000, 4), (20000, 5)]:
    tr = simular(0.0, n=n, semilla=sd)
    print("  n = %5d, semilla = %d -> %.4f" % (n, sd, (np.exp(tr[:, :, 2]).mean(axis=1) < UMBRAL).mean()))

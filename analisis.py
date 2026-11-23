"""Lógica de análisis del proyecto integrador (sin interfaz).
Series: TPM y tasa de colocación total (Banco de la República) y área aprobada para vivienda,
77 municipios (DANE, ELIC). Frecuencia mensual, 2005-01 a 2026-09.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
import os
_AQUI = os.path.dirname(os.path.abspath(__file__))
# Busca la base en data/ y, si no está, en la misma carpeta que este archivo (para repositorios con archivos sueltos)
RUTA = next((p for p in (os.path.join(_AQUI, "data", "base_integrador.xlsx"), os.path.join(_AQUI, "base_integrador.xlsx"))
             if os.path.exists(p)), "data/base_integrador.xlsx")
HOJA = "Base_Analizada"
LAGS = 3
ORDEN_ESTADOS = ["Bajo", "Medio", "Alto"]


# ------------------------------------------------------------------ datos
def cargar(ruta=RUTA):
    datos = pd.read_excel(ruta, sheet_name=HOJA)
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    df = datos.set_index("fecha")
    df.index = df.index.to_period("M")
    df["ln_area"] = np.log(df["area_vivienda_77mun_m2"])
    return df


def tabla_adf(df):
    base = df[["tpm", "tasa_colocacion_total", "ln_area"]]
    filas = []
    for c in base.columns:
        for etiqueta, s in ((" (nivel)", base[c]), (" (1ª diferencia)", base[c].diff())):
            r = adfuller(s.dropna(), regression="c", autolag="AIC")
            filas.append({"serie": c + etiqueta, "estadístico": round(r[0], 3),
                          "p-valor": round(r[1], 3), "rezagos": r[2], "n": r[3]})
    return pd.DataFrame(filas)


def estacionalidad(df):
    return df["ln_area"].groupby(df.index.month).mean().round(3)


# -------------------------------------------------------------------- VAR
def preparar_var(df):
    modelo = pd.DataFrame({"d_tpm": df["tpm"].diff(),
                           "d_tasa": df["tasa_colocacion_total"].diff(),
                           "ln_area": df["ln_area"]}).dropna()
    dic = pd.DataFrame({"dic": (modelo.index.month == 12).astype(float)}, index=modelo.index)
    modelo.index = modelo.index.to_timestamp()
    modelo = modelo.asfreq("MS")
    dic.index = modelo.index
    return modelo, dic


def criterios_rezagos(modelo, dic, maxlags=12):
    sel = VAR(modelo, exog=dic).select_order(maxlags=maxlags)
    tabla = pd.DataFrame(sel.ics).round(4)
    tabla.index.name = "rezagos"
    return tabla, {k: int(v) for k, v in sel.selected_orders.items()}


def estimar_var(modelo, dic, lags=LAGS):
    return VAR(modelo, exog=dic).fit(lags)


def tabla_granger(res, modelo):
    filas = []
    for causa in modelo.columns:
        for efecto in modelo.columns:
            if causa != efecto:
                t = res.test_causality(efecto, [causa], kind="f")
                filas.append({"causa": causa, "efecto": efecto,
                              "F": round(float(t.test_statistic), 3), "p-valor": round(float(t.pvalue), 4)})
    return pd.DataFrame(filas)


def diagnosticos(res):
    w = res.test_whiteness(nlags=12, adjusted=True)
    n = res.test_normality()
    return {"portmanteau": (float(w.test_statistic), int(w.df), float(w.pvalue)),
            "normalidad": (float(n.test_statistic), int(n.df), float(n.pvalue)),
            "raices": np.abs(res.roots), "estable": bool(res.is_stable())}


def irf_con_bandas(res, steps=12, repl=500, seed=2026):
    irf = res.irf(steps)
    bajo, alto = irf.errband_mc(orth=True, repl=repl, signif=0.05, seed=seed)
    return irf.orth_irfs, bajo, alto


def varianza(res, steps=12, nombres=None):
    f = res.fevd(steps).decomp[:, -1, :]
    return pd.DataFrame(f, index=nombres, columns=nombres).round(3)


def pronostico(res, modelo, df, h=12):
    fechas = pd.date_range(modelo.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    exog = (fechas.month == 12).astype(float).reshape(-1, 1)
    fc = pd.DataFrame(res.forecast(modelo.values[-LAGS:], steps=h, exog_future=exog),
                      index=fechas, columns=modelo.columns)
    return pd.DataFrame({"tpm": df["tpm"].iloc[-1] + fc["d_tpm"].cumsum(),
                         "tasa_colocacion_total": df["tasa_colocacion_total"].iloc[-1] + fc["d_tasa"].cumsum(),
                         "area_vivienda_m2": np.exp(fc["ln_area"])})


def evaluacion_fuera_muestra(modelo, dic, meses=12):
    corte = modelo.index[-1] - pd.DateOffset(months=meses)
    ent, prueba = modelo.loc[:corte], modelo.loc[corte + pd.offsets.MonthBegin(1):]
    r = VAR(ent, exog=dic.loc[ent.index]).fit(LAGS)
    f = pd.DataFrame(r.forecast(ent.values[-LAGS:], steps=len(prueba), exog_future=dic.loc[prueba.index].values),
                     index=prueba.index, columns=modelo.columns)
    rmse = lambda e: float(np.sqrt((e ** 2).mean()))
    filas = []
    for c in modelo.columns:
        ref = ent[c].iloc[-1] if c == "ln_area" else 0.0
        filas.append({"variable": c, "RMSE VAR": round(rmse(prueba[c] - f[c]), 3),
                      "RMSE ingenuo": round(rmse(prueba[c] - ref), 3)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------- Markov
def factor_estacional(df):
    ln = df["ln_area"]
    return ln.groupby(ln.index.month).transform("mean") - ln.mean()


def markov(df):
    ln = df["ln_area"]
    ln_adj = ln - factor_estacional(df)
    q1, q2 = ln_adj.quantile([1 / 3, 2 / 3])
    est = pd.cut(ln_adj, bins=[-np.inf, q1, q2, np.inf], labels=ORDEN_ESTADOS)
    o, d = est.iloc[:-1].values, est.iloc[1:].values
    conteo = (pd.crosstab(pd.Series(o, name="actual"), pd.Series(d, name="siguiente"))
              .reindex(index=ORDEN_ESTADOS, columns=ORDEN_ESTADOS, fill_value=0))
    P = conteo.div(conteo.sum(axis=1), axis=0)
    dur = pd.Series(1 / (1 - np.diag(P.values)), index=ORDEN_ESTADOS)
    vals, vecs = np.linalg.eig(P.values.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = pd.Series(pi / pi.sum(), index=ORDEN_ESTADOS)
    actual = str(est.iloc[-1])
    p0 = np.zeros(3)
    p0[ORDEN_ESTADOS.index(actual)] = 1
    probs = pd.DataFrame({f"{k} meses": p0 @ np.linalg.matrix_power(P.values, k) for k in (3, 6, 12)},
                         index=ORDEN_ESTADOS)
    m = len(o) // 2
    mitades = {}
    for nombre, oo, dd in (("2005-2015", o[:m], d[:m]), ("2015-2026", o[m:], d[m:])):
        c = (pd.crosstab(pd.Series(oo), pd.Series(dd))
             .reindex(index=ORDEN_ESTADOS, columns=ORDEN_ESTADOS, fill_value=0))
        mitades[nombre] = c.div(c.sum(axis=1), axis=0)
    return {"estados": est, "ln_adj": ln_adj, "q1": float(q1), "q2": float(q2), "conteo": conteo, "P": P,
            "duracion": dur, "pi": pi, "actual": actual, "probs": probs, "mitades": mitades}


# ------------------------------------------------------------ Monte Carlo
def montecarlo(res, modelo, df, mk, shock=0.10, n=10000, semilla=2026, h=12, pct_umbral=0.25, meses_shock=4):
    K = modelo.shape[1]
    A, c, bd, E = res.coefs, res.params.loc["const"].values, res.params.loc["dic"].values, res.resid.values
    fechas = pd.date_range(modelo.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    D = (fechas.month == 12).astype(float)
    area = df["area_vivienda_77mun_m2"]
    umbral = area.rolling(12).mean().dropna().quantile(pct_umbral)
    fac12 = factor_estacional(df).groupby(df.index.month).mean().loc[fechas[-1].month]

    def simular(s, nn, sd, ruido=True):
        rng = np.random.default_rng(sd)
        y = np.repeat(modelo.values[-LAGS:][None, :, :], nn, axis=0)
        tr = np.zeros((nn, h, K))
        for t in range(h):
            e = E[rng.integers(0, len(E), size=nn)] if ruido else 0.0
            mu = c + D[t] * bd
            for l in range(LAGS):
                mu = mu + y[:, -1 - l, :] @ A[l].T
            nuevo = mu + e
            if t < meses_shock:
                nuevo[:, 0] += s
            tr[:, t, :] = nuevo
            y = np.concatenate([y[:, 1:, :], nuevo[:, None, :]], axis=1)
        return tr

    chequeo = float(np.abs(simular(0.0, 1, 0, ruido=False)[0] -
                           res.forecast(modelo.values[-LAGS:], steps=h, exog_future=D.reshape(-1, 1))).max())
    escenarios = {f"Favorable (−{shock:.2f} pp/mes)": -shock, "Central": 0.0, f"Adverso (+{shock:.2f} pp/mes)": +shock}
    tpm0 = df["tpm"].iloc[-1]
    filas, trayectorias = [], {}
    for nombre, s in escenarios.items():
        tr = simular(s, n, semilla)
        am = np.exp(tr[:, :, 2])
        prom = am.mean(axis=1)
        p5, p50, p95 = np.percentile(prom, [5, 50, 95])
        bajo12 = ((tr[:, -1, 2] - fac12) <= mk["q1"]).mean()
        filas.append({"escenario": nombre, "P5": p5, "Mediana": p50, "P95": p95,
                      "P(prom < umbral)": (prom < umbral).mean(), "P(estado Bajo mes 12)": bajo12,
                      "TPM mes 12 (mediana)": np.median(tpm0 + tr[:, :, 0].sum(axis=1))})
        trayectorias[nombre] = (fechas, np.percentile(am, [5, 50, 95], axis=0))
    estab = []
    for nn, sd in [(2000, 1), (5000, 2), (10000, 3), (10000, 4), (20000, 5)]:
        tr = simular(0.0, nn, sd)
        estab.append({"n": nn, "semilla": sd, "P(prom < umbral)": float((np.exp(tr[:, :, 2]).mean(axis=1) < umbral).mean())})
    return {"tabla": pd.DataFrame(filas).set_index("escenario"), "umbral": float(umbral),
            "prom_ult12": float(area.iloc[-12:].mean()), "chequeo": chequeo,
            "estabilidad": pd.DataFrame(estab), "trayectorias": trayectorias,
            "p_bajo12_markov": float(mk["probs"].loc["Bajo", "12 meses"])}

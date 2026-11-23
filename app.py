"""App de Streamlit - Proyecto integrador: política monetaria, costo del crédito y actividad edificadora de vivienda."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analisis as an

AUTORA = "Maria Camila Acosta Puentes"
st.set_page_config(page_title="Crédito, política monetaria y vivienda", page_icon="🏠", layout="wide")


# ------------------------------------------------------------------ cálculo (en caché)
@st.cache_data
def cargar_todo():
    df = an.cargar()
    modelo, dic = an.preparar_var(df)
    res = an.estimar_var(modelo, dic)
    return {
        "df": df, "modelo": modelo, "dic": dic,
        "adf": an.tabla_adf(df), "est": an.estacionalidad(df),
        "crit": an.criterios_rezagos(modelo, dic), "params": res.params.round(4), "pvals": res.pvalues.round(4),
        "granger": an.tabla_granger(res, modelo), "diag": an.diagnosticos(res),
        "irf": an.irf_con_bandas(res), "fevd": an.varianza(res, nombres=list(modelo.columns)),
        "pron": an.pronostico(res, modelo, df), "oos": an.evaluacion_fuera_muestra(modelo, dic),
        "mk": an.markov(df),
    }


@st.cache_data
def correr_mc(shock, n, semilla, pct):
    df = an.cargar()
    modelo, dic = an.preparar_var(df)
    res = an.estimar_var(modelo, dic)
    mk = an.markov(df)
    return an.montecarlo(res, modelo, df, mk, shock=shock, n=n, semilla=semilla, pct_umbral=pct)


S = cargar_todo()
df, modelo = S["df"], S["modelo"]
NOMBRES = {"tpm": "TPM (%)", "tasa_colocacion_total": "Tasa de colocación total (%)",
           "area_vivienda_77mun_m2": "Área aprobada vivienda, 77 mun. (m²)", "ln_area": "ln(área aprobada)",
           "d_tpm": "Δ TPM", "d_tasa": "Δ tasa de colocación"}


def interpretar(muestra, significa, no_permite, siguiente=None):
    st.markdown(f"**Qué muestra.** {muestra}")
    st.markdown(f"**Qué significa para la pregunta.** {significa}")
    st.markdown(f"**Qué no permite concluir.** {no_permite}")
    if siguiente:
        st.markdown(f"**Siguiente paso.** {siguiente}")


# ------------------------------------------------------------------ navegación
st.sidebar.title("Navegación")
seccion = st.sidebar.radio("Sección", ["Inicio", "Datos", "VAR", "Markov", "Monte Carlo", "Conclusiones"])
st.sidebar.caption("Proyecto integrador - Análisis e Interpretación de Datos a través de Python")
st.sidebar.caption(f"Autora: {AUTORA}")
st.sidebar.warning("Ejercicio académico. Los resultados describen asociaciones y simulaciones condicionadas al modelo; "
                   "no son recomendaciones de inversión ni de política.")

# ================================================================== INICIO
if seccion == "Inicio":
    st.title("Política monetaria, costo del crédito y actividad edificadora de vivienda en Colombia")
    st.caption("Series de tiempo y VAR · Cadenas de Markov · Simulación de Monte Carlo")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Problema")
        st.write("Cuando el Banco de la República sube o baja su tasa de política (TPM), el cambio debería "
                 "trasladarse al costo del crédito y, después, a la actividad de la construcción de vivienda. "
                 "No es claro cuánto de esa transmisión se observa en los datos ni qué tan probable es que la "
                 "actividad edificadora quede en niveles bajos en los próximos meses.")
        st.subheader("Pregunta central")
        st.info("¿Cómo se relacionan dinámicamente la TPM, la tasa de colocación y el área aprobada para vivienda, "
                "y qué probabilidad hay de que la actividad edificadora caiga por debajo de un umbral crítico en "
                "los próximos 12 meses?")
    with c2:
        st.subheader("Usuario y decisión que apoya")
        st.write("Un planificador de oferta de vivienda o de colocación de crédito (constructora, entidad financiera) "
                 "que debe decidir cuándo ajustar su oferta frente a cambios en el costo del dinero.")
        st.subheader("Datos")
        st.write(f"{len(df)} observaciones mensuales, de {df.index.min()} a {df.index.max()}.")
    st.subheader("Variables y fuentes")
    st.dataframe(pd.DataFrame({
        "Variable": ["tpm", "tasa_colocacion_total", "area_vivienda_77mun_m2"],
        "Descripción": ["Tasa de política monetaria", "Tasa de interés de colocación, total (promedio mensual)",
                        "Área aprobada para vivienda en licencias de construcción, 77 municipios"],
        "Unidad": ["%", "%", "m²"],
        "Fuente": ["Banco de la República", "Banco de la República (datos de la Superfinanciera)",
                   "DANE, Estadísticas de Edificación - Licencias (ELIC), cobertura de 77 municipios"]}),
        hide_index=True)
    st.subheader("Cómo se resuelve")
    st.write("1) **Datos**: preparación y estacionariedad. 2) **VAR**: relaciones dinámicas y diagnósticos. "
             "3) **Markov**: estados Bajo, Medio y Alto de la actividad edificadora. "
             "4) **Monte Carlo**: 10.000 trayectorias a 12 meses y tres escenarios de TPM. "
             "5) **Conclusiones** y limitaciones.")
    st.caption("Aviso: no se incluye la tasa específica de crédito hipotecario, porque no se consiguió una serie mensual "
               "suficientemente larga. Se usa la tasa de colocación total como aproximación del costo del crédito.")

# ================================================================== DATOS
elif seccion == "Datos":
    st.title("Datos y preparación de las series")
    rango = st.slider("Periodo a mostrar", min_value=2005, max_value=2026, value=(2005, 2026))
    sub = df[(df.index.year >= rango[0]) & (df.index.year <= rango[1])]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[NOMBRES["tpm"], NOMBRES["tasa_colocacion_total"], NOMBRES["area_vivienda_77mun_m2"]])
    x = sub.index.to_timestamp()
    for i, c in enumerate(["tpm", "tasa_colocacion_total", "area_vivienda_77mun_m2"], start=1):
        fig.add_trace(go.Scatter(x=x, y=sub[c], mode="lines", showlegend=False), row=i, col=1)
    fig.update_layout(height=650, margin=dict(t=50, b=20))
    st.plotly_chart(fig)
    interpretar(
        "Las tasas muestran ciclos largos (descenso hasta 2013, subida en 2022-2023); el área aprobada es más volátil "
        "y tiene un pico cada diciembre.",
        "Las tasas son persistentes y el área responde a un ciclo propio, lo que justifica un modelo dinámico (VAR).",
        "El gráfico no prueba que una serie cause a otra.",
        "Probar estacionariedad y decidir transformaciones.")
    st.subheader("Estacionariedad (prueba ADF)")
    st.dataframe(S["adf"], hide_index=True)
    st.caption("H0: raíz unitaria (constante incluida; rezagos por AIC). La TPM en nivel tiene p ≈ 0,051, justo en el "
               "límite; con una sola prueba no se acepta ni se descarta una raíz unitaria.")
    st.markdown("**Decisión:** tasas en primera diferencia (Δ TPM, Δ tasa de colocación); el área en logaritmo y en nivel "
                "(ya estacionaria); dummy de diciembre como variable exógena.")
    st.subheader("Estacionalidad del área aprobada")
    est = S["est"]
    fe = go.Figure(go.Bar(x=list(est.index), y=list(est.values)))
    fe.update_layout(height=300, xaxis_title="Mes", yaxis_title="Promedio de ln(área)", margin=dict(t=10))
    fe.update_yaxes(range=[13.6, 14.3])
    st.plotly_chart(fe)
    st.subheader("Datos")
    st.dataframe(df[["tpm", "tasa_colocacion_total", "tasa_consumo", "area_vivienda_77mun_m2"]].round(3))
    st.download_button("Descargar base (CSV)", df.drop(columns="ln_area").to_csv().encode("utf-8"),
                       "base_integrador.csv", "text/csv")

# ================================================================== VAR
elif seccion == "VAR":
    st.title("Modelo VAR")
    st.write("VAR de tres variables mensuales: **Δ TPM, Δ tasa de colocación y ln(área aprobada)**, con dummy de "
             "diciembre como exógena. Muestra 2005-02 a 2026-09.")
    tab = st.tabs(["Rezagos y estabilidad", "Granger", "Impulso-respuesta", "Varianza", "Pronóstico"])
    with tab[0]:
        crit, sel = S["crit"]
        st.dataframe(crit)
        st.write(f"Rezagos sugeridos: {sel}. Se usa **VAR(3)**.")
        d = S["diag"]
        st.write(f"**Estabilidad:** {'estable' if d['estable'] else 'NO estable'} (raíz más cercana al círculo unitario: "
                 f"{d['raices'].min():.3f}, debe ser > 1).")
        st.write(f"**Autocorrelación residual (Portmanteau, 12 rezagos):** estadístico {d['portmanteau'][0]:.2f}, "
                 f"p = {d['portmanteau'][2]:.4f} → no se rechaza ruido blanco.")
        st.write(f"**Normalidad (Jarque-Bera):** p = {d['normalidad'][2]:.4f} → se rechaza; los residuos tienen colas pesadas.")
        st.markdown("**Coeficientes estimados (p-valores entre paréntesis en la segunda tabla)**")
        st.dataframe(S["params"])
        st.dataframe(S["pvals"])
        interpretar(
            "Un VAR(3) estable cuyos residuos no muestran autocorrelación significativa, aunque no son normales.",
            "La dinámica conjunta está razonablemente capturada; los intervalos y pruebas F son aproximados.",
            "Los coeficientes individuales no se interpretan como efectos causales.")
    with tab[1]:
        st.dataframe(S["granger"], hide_index=True)
        interpretar(
            "Los cambios de la TPM ayudan a predecir los cambios de la tasa de colocación (p ≈ 0). La TPM no ayuda a "
            "predecir el área (p ≈ 0,63) ni la tasa de colocación al área (p ≈ 0,14). El área sí predice la TPM y la tasa.",
            "La transmisión TPM → costo del crédito se ve en los datos; su llegada a la actividad edificadora, no.",
            "Granger mide poder predictivo condicionado al modelo, no causalidad estructural. Que el área 'prediga' la TPM "
            "refleja probablemente el ciclo económico común o variables omitidas.")
    with tab[2]:
        orth, bajo, alto = S["irf"]
        nombres = list(modelo.columns)
        imp = st.selectbox("Choque en", nombres, index=0, format_func=lambda v: NOMBRES[v])
        i = nombres.index(imp)
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"Respuesta de {NOMBRES[n]}" for n in nombres])
        h = np.arange(orth.shape[0])
        for j in range(3):
            fig.add_trace(go.Scatter(x=h, y=alto[:, j, i], line=dict(width=0), showlegend=False), row=1, col=j + 1)
            fig.add_trace(go.Scatter(x=h, y=bajo[:, j, i], fill="tonexty", line=dict(width=0),
                                     fillcolor="rgba(120,120,120,0.25)", showlegend=False), row=1, col=j + 1)
            fig.add_trace(go.Scatter(x=h, y=orth[:, j, i], mode="lines", showlegend=False), row=1, col=j + 1)
            fig.add_hline(y=0, line_dash="dot", row=1, col=j + 1)
        fig.update_layout(height=380, margin=dict(t=50))
        fig.update_xaxes(title_text="Meses")
        st.plotly_chart(fig)
        st.caption("Ordenamiento de Cholesky: Δ TPM, Δ tasa de colocación, ln(área). Choque de 1 desviación estándar; "
                   "banda al 95 % por simulación (500 réplicas, semilla 2026). Supone que la TPM no responde "
                   "contemporáneamente a las otras variables; con otro orden los resultados cambian.")
        incluye = bool(((bajo[1:, 2, 0] <= 0) & (alto[1:, 2, 0] >= 0)).all())
        interpretar(
            "Ante un choque en Δ TPM, la respuesta de la tasa de colocación es positiva y la del área es pequeña"
            + (" y su banda incluye el cero en todos los meses." if incluye else "."),
            "No hay evidencia de un efecto sistemático de la TPM sobre el área en estos datos.",
            "Un efecto no significativo no prueba que no exista; puede deberse al rezago, a la tasa elegida o a la muestra.")
    with tab[3]:
        st.dataframe(S["fevd"])
        st.caption("Filas: variable explicada; columnas: choque que la explica, a 12 meses. El área se explica en 90 % por su "
                   "propia historia; la TPM explica solo 6 %. La tasa de colocación se explica en 28 % por la TPM.")
    with tab[4]:
        pr = S["pron"]
        st.dataframe(pr.round(2))
        fig = go.Figure()
        hist = df["tpm"].iloc[-36:]
        fig.add_trace(go.Scatter(x=hist.index.to_timestamp(), y=hist.values, name="TPM observada"))
        fig.add_trace(go.Scatter(x=pr.index, y=pr["tpm"], name="TPM pronosticada", line=dict(dash="dash")))
        fig.update_layout(height=320, margin=dict(t=10), yaxis_title="%")
        st.plotly_chart(fig)
        st.markdown("**Evaluación fuera de muestra (12 meses)**")
        st.dataframe(S["oos"], hide_index=True)
        st.warning("El VAR apenas supera al pronóstico ingenuo en las tasas y es peor en el área. El pronóstico de la TPM "
                   "sube por la inercia del último cambio y no debe leerse como predicción de política monetaria. "
                   "El modelo sirve para entender relaciones, no como herramienta de pronóstico.")

# ================================================================== MARKOV
elif seccion == "Markov":
    st.title("Cadena de Markov de la actividad edificadora")
    mk = S["mk"]
    st.write("**Qué cambia de estado:** el área aprobada para vivienda, desestacionalizada (se resta el promedio de cada mes). "
             "**Estados:** terciles de la serie (Bajo, Medio, Alto). **Intervalo:** un mes.")
    st.write(f"Umbrales (m², desestacionalizados): Bajo ≤ {np.exp(mk['q1']):,.0f} · Medio ≤ {np.exp(mk['q2']):,.0f} · "
             f"Alto > {np.exp(mk['q2']):,.0f}. Estado actual ({df.index[-1]}): **{mk['actual']}**.")
    colores = {"Bajo": "#c0392b", "Medio": "#e1a100", "Alto": "#2e8b57"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mk["ln_adj"].index.to_timestamp(), y=np.exp(mk["ln_adj"]), mode="lines",
                             line=dict(color="lightgray"), showlegend=False))
    for e in an.ORDEN_ESTADOS:
        m = mk["estados"] == e
        fig.add_trace(go.Scatter(x=mk["ln_adj"].index[m].to_timestamp(), y=np.exp(mk["ln_adj"][m]), mode="markers",
                                 name=e, marker=dict(color=colores[e], size=5)))
    fig.update_layout(height=340, margin=dict(t=10), yaxis_title="m² desestacionalizados")
    st.plotly_chart(fig)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Conteo de transiciones** (fila = estado actual, columna = siguiente)")
        st.dataframe(mk["conteo"])
    with c2:
        st.markdown("**Matriz de transición** (cada fila suma 1)")
        st.dataframe(mk["P"].round(3))
    st.write(f"Suma de filas: {mk['P'].sum(axis=1).round(6).tolist()}")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Duración esperada (meses)**")
        st.dataframe(mk["duracion"].round(2).rename("meses"))
    with c4:
        st.markdown("**Probabilidades desde el estado actual**")
        st.dataframe(mk["probs"].round(3))
    interpretar(
        "Los estados Bajo y Alto son persistentes (≈ 64 % y 66 % de permanencia); Medio es el más inestable. Pasar de Bajo "
        "a Alto en un mes es raro (≈ 5 %). Desde el estado actual (Bajo), la probabilidad de seguir en Bajo baja de ≈ 43 % a "
        "3 meses a ≈ 33 % a 12 meses, que coincide con la distribución estacionaria.",
        "La actividad edificadora tiende a permanecer varios meses en un mismo nivel; recuperarse desde Bajo suele tomar "
        "tiempo, pero a un año la cadena ya no 'recuerda' el punto de partida.",
        "La matriz describe cómo se mueve el área, no por qué. No incluye el efecto de las tasas.")
    st.subheader("Homogeneidad temporal")
    c5, c6 = st.columns(2)
    for col, nombre in zip((c5, c6), mk["mitades"]):
        with col:
            st.markdown(f"**Matriz {nombre}**")
            st.dataframe(mk["mitades"][nombre].round(2))
    st.warning("Las dos mitades de la muestra no tienen la misma matriz y los estados se concentran por épocas "
               "(predomina Bajo en 2005-2009, Alto en 2010-2014 y Bajo otra vez en 2020-2024). El supuesto de homogeneidad "
               "temporal es débil; los terciles mezclan ciclos largos con movimientos de corto plazo.")

# ================================================================== MONTE CARLO
elif seccion == "Monte Carlo":
    st.title("Simulación de Monte Carlo")
    st.write("Se simulan trayectorias a 12 meses (oct-2026 a sep-2027) con el VAR(3), remuestreando los residuos del "
             "propio modelo (bootstrap) para conservar su correlación. Los escenarios imponen un choque sobre Δ TPM "
             "durante los primeros 4 meses; el VAR lo amplifica por la inercia de la TPM.")
    with st.sidebar:
        st.subheader("Parámetros de la simulación")
        n = st.select_slider("Trayectorias", [2000, 5000, 10000, 20000], value=10000)
        semilla = st.number_input("Semilla", value=2026, step=1)
        shock = st.slider("Choque en Δ TPM (pp por mes, 4 meses)", 0.0, 0.25, 0.10, 0.05)
        pct = st.slider("Umbral crítico (percentil del promedio móvil de 12 meses)", 0.10, 0.40, 0.25, 0.05)
    mc = correr_mc(shock, int(n), int(semilla), pct)
    st.write(f"**Umbral crítico:** {mc['umbral']:,.0f} m²/mes (percentil {int(pct * 100)} del promedio móvil de 12 meses "
             f"histórico). Promedio de los últimos 12 meses observados: {mc['prom_ult12']:,.0f} m²/mes. "
             f"Chequeo (simulación sin ruido vs pronóstico del VAR): diferencia máxima = {mc['chequeo']:.1e}.")
    t = mc["tabla"].copy()
    fmt = {"P5": "{:,.0f}", "Mediana": "{:,.0f}", "P95": "{:,.0f}", "P(prom < umbral)": "{:.3f}",
           "P(estado Bajo mes 12)": "{:.3f}", "TPM mes 12 (mediana)": "{:.2f}"}
    st.dataframe(t.style.format(fmt))
    fig = go.Figure()
    col = ["#2e8b57", "#555555", "#c0392b"]
    for (nombre, (fechas, p)), c in zip(mc["trayectorias"].items(), col):
        fig.add_trace(go.Scatter(x=fechas, y=p[1], name=nombre, line=dict(color=c)))
        fig.add_trace(go.Scatter(x=list(fechas) + list(fechas[::-1]), y=list(p[2]) + list(p[0][::-1]), fill="toself",
                                 fillcolor="rgba(150,150,150,0.10)", line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
    fig.add_hline(y=mc["umbral"], line_dash="dot", annotation_text="umbral crítico")
    fig.update_layout(height=420, yaxis_title="Área mensual (m²): mediana y banda P5-P95", margin=dict(t=10))
    st.plotly_chart(fig)
    st.markdown("**Estabilidad numérica** (escenario central, distintas semillas y tamaños)")
    st.dataframe(mc["estabilidad"].style.format({"P(prom < umbral)": "{:.4f}"}), hide_index=True)
    st.write(f"Referencia de Markov: P(estado Bajo a 12 meses desde el estado actual) = {mc['p_bajo12_markov']:.3f}.")
    interpretar(
        "En el escenario central, la probabilidad de que el área promedio de los próximos 12 meses quede por debajo del "
        "umbral es de ≈ 26 % (con los parámetros por defecto), y la de terminar en estado Bajo al mes 12 es de ≈ 47 %. La "
        "banda P5-P95 es muy amplia.",
        "El riesgo de baja actividad no es marginal, pero la incertidumbre es grande frente a las diferencias entre escenarios.",
        "Los escenarios son condicionales al modelo, no predicciones. Resultado contraintuitivo: el escenario adverso (TPM "
        "más alta) sale con más área, porque en el VAR el efecto de la TPM sobre el área no es significativo y el de la tasa de "
        "colocación tiene signo positivo (las tasas suben en épocas de auge). No debe leerse como 'subir tasas favorece "
        "la construcción'.")

# ================================================================== CONCLUSIONES
elif seccion == "Conclusiones":
    st.title("Conclusiones y limitaciones")
    st.subheader("Respuesta a la pregunta central")
    st.markdown(
        "- **Transmisión al costo del crédito:** los cambios de la TPM predicen los de la tasa de colocación (Granger, "
        "p ≈ 0) y la TPM explica ≈ 28 % de la varianza de la tasa a 12 meses.\n"
        "- **Transmisión a la actividad edificadora:** no se observa. Ni la TPM ni la tasa de colocación predicen el área "
        "de forma significativa, y la respuesta del área a un choque en la TPM tiene bandas que incluyen el cero.\n"
        "- **Riesgo de baja actividad:** el Markov y el Monte Carlo coinciden en que la probabilidad de estar en niveles "
        "bajos a 12 meses ronda entre 33 % y 47 %, según la definición; no es marginal.")
    st.subheader("Para la decisión del usuario")
    st.write("El costo del crédito sí responde a la política monetaria, pero la actividad de licencias de vivienda se "
             "explica sobre todo por su propia dinámica y por ciclos largos. Quien planee oferta debe vigilar la actividad "
             "reciente y no solo la TPM. Este análisis no es una recomendación de política ni de inversión.")
    st.subheader("Limitaciones")
    st.markdown(
        "- No se incluyó la tasa de crédito **hipotecario**: se usa la tasa de colocación total. La financiación de "
        "vivienda del DANE (FIVI) solo cubre 2022-2026 (19 trimestres) y no alcanza para un VAR.\n"
        "- El área aprobada mide **licencias**, no obra terminada ni ventas.\n"
        "- Los residuos del VAR **no son normales** (colas pesadas por 2008 y 2020); los intervalos son aproximados.\n"
        "- El **orden de Cholesky** supone que la TPM no responde el mismo mes a las otras variables.\n"
        "- La TPM en nivel tiene p ≈ 0,051 en la prueba ADF; se tratan en diferencias por prudencia.\n"
        "- La **cadena de Markov no es homogénea** en el tiempo y los terciles mezclan ciclos largos con movimientos cortos.\n"
        "- El **pronóstico** del VAR apenas supera al ingenuo; sirve para entender relaciones, no para predecir.\n"
        "- Asociación y causalidad de Granger **no** equivalen a causalidad estructural.")
    st.subheader("Bibliografía")
    st.markdown(
        "- Melo, L. F. y Hamann, F. A. *Inflación básica: una estimación basada en modelos VAR estructurales.* Banco de la República.\n"
        "- Rosas Rosas, L. C. y Segundo Rosas, J. I. (2019). Sobre cadenas de Markov y su potencial en las aplicaciones. *Epistemus*, 26, 7-12.\n"
        "- Romero, P. y Rueda, R. (2013). *Ars Conjectandi* y el método Monte Carlo. *Revista Digital Universitaria*, 14(11), UNAM.\n"
        "- Hyndman, R. J. y Athanasopoulos, G. *Forecasting: Principles and Practice.*\n"
        "- Lütkepohl, H. *New Introduction to Multiple Time Series Analysis.*\n"
        "- Ross, S. M. *Introduction to Probability Models.*\n"
        "- Documentación oficial de pandas, Plotly y Streamlit.\n"
        "- Datos: Banco de la República (TPM, tasas de colocación); DANE (licencias de construcción, ELIC).")
    st.subheader("Registro de uso de IA")
    st.write("Se usó un asistente de IA para estructurar el análisis, redactar el código y revisar la interpretación. "
             "Todos los resultados fueron ejecutados, verificados y revisados por la autora con los datos reales; ver el "
             "registro detallado en el informe final.")

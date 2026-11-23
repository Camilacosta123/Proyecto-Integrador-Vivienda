# Política monetaria, costo del crédito y actividad edificadora de vivienda en Colombia

Proyecto integrador de *Análisis e Interpretación de Datos a través de Python* (tercer corte):
series de tiempo y modelo VAR, cadenas de Markov y simulación de Monte Carlo, presentados en una app de Streamlit.

**Autora:** [Nombre de la estudiante]
**App publicada:** [pegar aquí el enlace de Streamlit Community Cloud]

## Pregunta central
¿Cómo se relacionan dinámicamente la tasa de política monetaria (TPM), la tasa de colocación y el área aprobada para
vivienda, y qué probabilidad hay de que la actividad edificadora caiga por debajo de un umbral crítico en los próximos 12 meses?

## Datos (mensuales, 2005-01 a 2026-09, 261 observaciones)
| Variable | Fuente | Unidad |
|---|---|---|
| `tpm` - Tasa de política monetaria | Banco de la República | % |
| `tasa_colocacion_total` - Tasa de colocación total (promedio mensual) | Banco de la República (datos de la Superfinanciera) | % |
| `area_vivienda_77mun_m2` - Área aprobada para vivienda, 77 municipios | DANE, Licencias de construcción (ELIC) | m² |
| `tasa_consumo` (apoyo, no entra al VAR) | Banco de la República | % |

Archivo: `data/base_integrador.xlsx`, hoja `Base_Analizada` (hoja `Diccionario` con unidades y fuentes).
Se usa la cobertura de 77 municipios porque es la única con el mismo universo durante todo el periodo.

## Contenido del repositorio
```
app.py                          # app de Streamlit (Inicio, Datos, VAR, Markov, Monte Carlo, Conclusiones)
analisis.py                     # lógica de análisis (carga, ADF, VAR, Markov, Monte Carlo)
proyecto_integrador_completo.py # script único con todo el análisis y sus salidas por consola
requirements.txt                # dependencias
data/base_integrador.xlsx       # base de datos (hoja Base_Analizada y hoja Diccionario)
data/fuentes_originales/        # archivos originales de BanRep y DANE, sin modificar (ver LEEME.txt)
figuras/                        # gráficos generados por el script
informe/                        # informe final en PDF
```

## Cómo ejecutarlo en local
```bash
python -m venv venv
venv\Scripts\activate            # Windows (en Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py             # la app
python proyecto_integrador_completo.py   # solo el análisis por consola
```
Ejecutar siempre desde la carpeta del repositorio (las rutas son relativas).

## Decisiones metodológicas
- Tasas en primera diferencia (la TPM en nivel da p ≈ 0,051 en ADF); área en logaritmo y en nivel; dummy de diciembre como exógena.
- VAR(3): AIC, BIC, HQIC y FPE eligen 3 rezagos; estable; sin autocorrelación residual (Portmanteau p = 0,12); residuos no normales.
- Markov: tres estados (terciles del área desestacionalizada); matriz con filas = estado actual.
- Monte Carlo: bootstrap de residuos del VAR, 10.000 trayectorias, semilla 2026, tres escenarios de TPM; chequeo sin ruido = pronóstico del VAR.

## Limitaciones
No incluye la tasa de crédito hipotecario (serie mensual larga no disponible); el área aprobada mide licencias, no obra
terminada; residuos no normales; orden de Cholesky supuesto; cadena de Markov no homogénea; el pronóstico del VAR apenas
supera al ingenuo; Granger no es causalidad estructural. Es un ejercicio académico, no una recomendación de política ni de inversión.

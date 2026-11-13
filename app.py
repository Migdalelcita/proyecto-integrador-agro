import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de página de Streamlit
st.set_page_config(page_title="Dashboard Agrotecnológico", layout="wide")
st.title("🌱 Optimización Inferencial del Rendimiento Agrícola")

# ==============================================================================
# FASE 1: TELEMETRÍA AGROINDUSTRIAL EN DUCKDB
# ==============================================================================
@st.cache_data
def generar_datos_telemetria():
    con = duckdb.connect(database=':memory:')
    
    # Paso 1.1: Fijar semilla determinista en DuckDB
    con.execute("SELECT setseed(0.42);")
    
    # Paso 1.2: Simulación SQL de N=100 parcelas según el algoritmo
    query_simulacion = """
    CREATE TABLE parcelas AS
    WITH base AS (
        SELECT 
            range AS parcela_id,
            ROUND(100.0 + random() * (300.0 - 100.0), 2) AS volumen_riego,
            ROUND(50.0 + random() * (150.0 - 50.0), 2) AS fertilizante_kg,
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            -0.8 + random() * (0.8 - (-0.8)) AS ruido
        FROM range(1, 101)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(
            5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + 
            CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END + 
            ruido, 
        2) AS rendimiento_ton
    FROM base;
    """
    con.execute(query_simulacion)
    
    # Paso 1.3: Extracción SQL a DataFrame de Pandas
    df = con.execute("SELECT * FROM parcelas").df()
    con.close()
    return df

df_raw = generar_datos_telemetria()

# ==============================================================================
# CONTROLES INTERACTIVOS (SIDEBAR - FASE 5)
# ==============================================================================
st.sidebar.header("Filtros del Invernadero")
tratamiento_sel = st.sidebar.multiselect(
    "Seleccionar Tratamiento:",
    options=df_raw["tipo_tratamiento"].unique(),
    default=df_raw["tipo_tratamiento"].unique()
)

riego_min, riego_max = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    float(df_raw["volumen_riego"].min()),
    float(df_raw["volumen_riego"].max()),
    (float(df_raw["volumen_riego"].min()), float(df_raw["volumen_riego"].max()))
)

# Aplicar filtros al DataFrame
df = df_raw[
    (df_raw["tipo_tratamiento"].isin(tratamiento_sel)) &
    (df_raw["volumen_riego"].between(riego_min, riego_max))
].copy()

if len(df) < 5:
    st.warning("Selecciona más datos en los filtros para calcular los modelos estadísticos.")
    st.stop()

# ==============================================================================
# FASE 2: REGRESIÓN LINEAL OLS Y CORRELACIÓN (SUBTEMA 4.2)
# ==============================================================================
# Matriz de correlación de Pearson
corr_matrix = df[["volumen_riego", "fertilizante_kg", "rendimiento_ton"]].corr(method="pearson")

# Modelo OLS Múltiple
X = df[["volumen_riego", "fertilizante_kg"]]
X = sm.add_constant(X) # Agregar intercepto B0
y = df["rendimiento_ton"]
modelo_ols = sm.OLS(y, X).fit()

r2_ajustado = modelo_ols.rsquared_adj
residuos = modelo_ols.resid
rmse = np.sqrt(np.mean(residuos**2))

# ==============================================================================
# FASE 3: PRUEBA DE HIPÓTESIS T-STUDENT (SUBTEMA 4.3)
# ==============================================================================
grupo_a = df_raw[df_raw["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"]
grupo_b = df_raw[df_raw["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"]

# Prueba t unilateral (H1: mu_B > mu_A)
t_stat, p_valor = stats.ttest_ind(grupo_b, grupo_a, alternative='greater', equal_var=False)

# Intervalo de confianza al 95% para la diferencia de medias (B - A)
diff_medias = grupo_b.mean() - grupo_a.mean()
se_diff = np.sqrt(grupo_b.var(ddof=1)/len(grupo_b) + grupo_a.var(ddof=1)/len(grupo_a))
grados_libertad = len(grupo_b) + len(grupo_a) - 2
ic_95 = stats.t.interval(0.95, df=grados_libertad, loc=diff_medias, scale=se_diff)

# ==============================================================================
# FASE 5: TARJETAS KPI Y MAQUETACIÓN WEB (2x2)
# ==============================================================================
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Rendimiento Promedio (Filtro)", f"{df['rendimiento_ton'].mean():.2f} ton/ha")
col_kpi2.metric("R² Ajustado del Modelo OLS", f"{r2_ajustado * 100:.1f}%")
col_kpi3.metric("P-Valor Prueba t (B > A)", f"{p_valor:.4e}", delta="Significativo" if p_valor < 0.05 else "No Significativo")

st.divider()

# Cuadrícula 2x2 para las 4 gráficas obligatorias (Fase 4)
fila1_col1, fila1_col2 = st.columns(2)
fila2_col1, fila2_col2 = st.columns(2)

# Gráfica 1: Mapa de Calor de Correlación
with fila1_col1:
    st.subheader("1. Matriz de Correlación de Pearson")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

# Gráfica 2: Dispersión con Línea de Regresión OLS
with fila1_col2:
    st.subheader("2. Regresión OLS: Riego vs. Rendimiento")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", ax=ax2, scatter_kws={"alpha":0.6}, line_kws={"color":"red"})
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig2)

# Gráfica 3: Diagrama de Cajas Comparativo
with fila2_col1:
    st.subheader("3. Comparativa de Tratamientos (A vs. B)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_raw, x="tipo_tratamiento", y="rendimiento_ton", palette="Set2", ax=ax3)
    ax3.set_xlabel("Tipo de Tratamiento")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig3)

# Gráfica 4: Histograma y Densidad de Residuos
with fila2_col2:
    st.subheader("4. Diagnóstico de Normalidad de Residuos")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuos, kde=True, color="purple", ax=ax4)
    ax4.axvline(0, color="black", linestyle="--")
    ax4.set_xlabel("Residuos (Real - Predicho)")
    st.pyplot(fig4)

# ==============================================================================
# RESUMEN TÉCNICO PARA EL REPORTE / SUSTENTACIÓN
# ==============================================================================
with st.expander("Ver Detalles Estadísticos (OLS y Prueba t)"):
    st.write(f"**Veredicto Inferencial:** Diferencia de medias = +{diff_medias:.2f} ton/ha | IC 95%: ({ic_95[0]:.2f}, {ic_95[1]:.2f})")
    if p_valor < 0.05:
        st.success("Se rechaza H0 (p < 0.05). Se recomienda escalar el Tratamiento_B.")
    else:
        st.info("No se rechaza H0. No hay evidencia suficiente de mejora.")
    st.text(modelo_ols.summary())

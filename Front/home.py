import streamlit as st
from pathlib import Path
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mínima acción - Cazando exoplanetas", layout="wide")
st.markdown(
    """
    <style>
    :root{
        --card-bg: #f7f7f9;
        --card-border: #e6e6e6;
        --card-text: #111827;
        --muted: #6c757d;
        --hero-text: #0f172a;
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        :root{
            --card-bg: #0f1724;        /* card dark bg */
            --card-border: #24303b;    /* subtle border */
            --card-text: #e6eef8;      /* light text */
            --muted: #9aa6b2;
            --hero-text: #ffffff;
        }
    }

    .hero { font-size:30px; font-weight:700; color:var(--hero-text); margin-bottom: 8px; }
    .subtitle { font-size:18px; color:var(--muted); margin-bottom: 24px; }
    .muted { color: var(--muted); }
    .card {
        background-color: var(--card-bg);
        padding:12px;
        border-radius:8px;
        border:1px solid var(--card-border);
        color: var(--card-text);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-value { font-size:26px; font-weight:700; color: var(--card-text); }
    .section-header { 
        font-size: 22px; 
        font-weight: 600; 
        margin-top: 32px; 
        margin-bottom: 16px;
        color: var(--hero-text);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Header / Hero ---
st.markdown('<div class="hero">Mínima Acción – Cazando exoplanetas</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Un vistazo visual a los candidatos y sus propiedades</div>', unsafe_allow_html=True)

# --- Importance text ---
st.markdown("#### ¿Por qué importa esto?")
st.write(
    "La búsqueda de exoplanetas requiere analizar grandes volúmenes de señales de tránsito y "
    "detectar patrones sutiles en datos fotométricos y orbitales. Un flujo reproducible que "
    "transforme los datos, aplique modelos de clasificación y visualice resultados facilita tanto "
    "la exploración científica como la validación rápida de nuevos candidatos."
)

st.divider()

# --- Load CSV (ruta relativa desde Front/) ---
DATA_PATH = Path(__file__).resolve().parent.parent.joinpath("Back", "predicciones.csv")
st.markdown(f"**Fuente de datos:** `{DATA_PATH}`")

df = None
if DATA_PATH.exists():
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        st.success(f"CSV cargado: {len(df):,} filas, {len(df.columns):,} columnas")
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
else:
    st.warning("No se encontró `Back/predicciones.csv`. Genera el CSV y colócalo en Back/.")


# -------------------------
# Modelo / Métricas (visual)
# -------------------------
st.markdown('<div class="section-header">Resumen del modelo y métricas</div>', unsafe_allow_html=True)

# Leaderboard
leaderboard_df = pd.DataFrame({
    "model": [
        "WeightedEnsemble_L2",
        "CatBoost_BAG_L1/T4",
        "CatBoost_BAG_L1/T1"
    ],
    "score_test": [0.985270, 0.984847, 0.984673],
    "score_val": [0.979885, 0.978945, 0.978919],
    "fit_time": [46.082008, 14.649281, 5.968385],
    "stack_level": [2, 1, 1],
})

# Key metrics
roc_auc = 0.9853
pr_auc = 0.9653
tn, fp, fn, tp = 1981, 65, 93, 731
accuracy = (tp + tn) / (tn + fp + fn + tp)
support_total = tn + fp + fn + tp

# Layout: leaderboard + metric cards
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("**Leaderboard (top modelos)**")
    st.dataframe(
        leaderboard_df.sort_values("score_test", ascending=False).reset_index(drop=True).style.format({
            "score_test": "{:.6f}",
            "score_val": "{:.6f}",
            "fit_time": "{:.1f}"
        }),
        hide_index=True
    )

with col2:
    st.write("")
    mcol1, mcol2 = st.columns(2)
    st.write("")
    mcol3, mcol4 = st.columns(2) 
    mcol1.markdown('<div class="card"><div style="font-size:12px;color:#6c757d">ROC-AUC (eval metric)</div>'
                   f'<div class="metric-value">{roc_auc:.4f}</div></div>', unsafe_allow_html=True)
    mcol2.markdown('<div class="card"><div style="font-size:12px;color:#6c757d">PR-AUC</div>'
                   f'<div class="metric-value">{pr_auc:.4f}</div></div>', unsafe_allow_html=True)
    mcol3.markdown('<div class="card"><div style="font-size:12px;color:#6c757d">Falsos Negativos</div>'
                   f'<div class="metric-value">{fn:,}</div></div>', unsafe_allow_html=True)
    mcol4.markdown('<div class="card"><div style="font-size:12px;color:#6c757d">Falsos Positivos</div>'
                   f'<div class="metric-value">{fp:,}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Nota:** el predictor fue entrenado con `eval_metric='roc_auc'` para maximizar ROC-AUC.")

st.divider()

st.markdown('<div class="section-header">Top features (importancia)</div>', unsafe_allow_html=True)

# Top features
fi_df = pd.DataFrame({
    "feature": ["koi_model_snr", "koi_count", "koi_prad", "duration_anomaly", "koi_dicco_mdec"],
    "importance": [0.039159, 0.004997, 0.004494, 0.003061, 0.002622]
}).sort_values("importance", ascending=True)

# Gráfico de barras horizontales (Altair)
bar = alt.Chart(fi_df).mark_bar(color="#1f77b4").encode(
    x=alt.X("importance:Q", title="Importancia"),
    y=alt.Y("feature:N", sort='-x', title="Feature"),
    tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".6f")]
).properties(height=220)
st.altair_chart(bar, use_container_width=True)

st.divider()

st.markdown('<div class="section-header">Exploración de resultados visual</div>', unsafe_allow_html=True)

if df is None:
    st.info("Sube o genera `Back/predicciones.csv` y recarga la página para ver las gráficas.")
else:
    st.markdown("**Kepler Magnitude vs G-band Magnitude**")

    x_col = "koi_kepmag"
    y_col = "koi_gmag"
    color_col = "koi_steff"
    size_col = "koi_model_snr"

    missing = [c for c in (x_col, y_col) if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas para la gráfica: {missing}.")
    else:
        plot_df = df[[x_col, y_col]].copy()
        for c in (color_col, size_col, "kepid", "target", "prediction"):
            if c in df.columns:
                plot_df[c] = df[c]
        plot_df = plot_df.dropna(subset=[x_col, y_col])

        tooltip_fields = ["kepid", x_col, y_col]
        if color_col in plot_df.columns:
            tooltip_fields.append(color_col)
        if size_col in plot_df.columns:
            tooltip_fields.append(size_col)
        if "target" in plot_df.columns:
            tooltip_fields.append("target")
        if "prediction" in plot_df.columns:
            tooltip_fields.append("prediction")

        base = alt.Chart(plot_df).mark_circle(opacity=0.9, size=80).encode(
            x=alt.X(f"{x_col}:Q", title="Kepler Magnitude"),
            y=alt.Y(f"{y_col}:Q", title="G-band Magnitude"),
            tooltip=[alt.Tooltip(f"{t}", format=".4f") if plot_df[t].dtype.kind in "fi" else alt.Tooltip(f"{t}") for t in tooltip_fields]
        )

        if color_col in plot_df.columns:
            chart = base.encode(color=alt.Color(f"{color_col}:Q", title="Teff (K)", scale=alt.Scale(scheme="viridis")))
        else:
            chart = base

        chart = chart.properties(width=900, height=500).interactive()
        st.altair_chart(chart, use_container_width=True)

    st.write("")

    st.markdown("**Spatial Distribution (RA vs Dec)**")

    ra_col = "ra"
    dec_col = "dec"
    color_spatial = "target"
    size_spatial = "koi_model_snr"

    missing_sp = [c for c in (ra_col, dec_col) if c not in df.columns]
    if missing_sp:
        st.error(f"No se puede mostrar la distribución espacial; faltan columnas: {missing_sp}")
    else:
        sp_df = df[[ra_col, dec_col]].copy()
        sp_df[ra_col] = pd.to_numeric(sp_df[ra_col], errors="coerce")
        sp_df[dec_col] = pd.to_numeric(sp_df[dec_col], errors="coerce")

        for c in ("kepid", color_spatial, size_spatial, "prediction"):
            if c in df.columns:
                sp_df[c] = df[c]

        sp_df = sp_df.dropna(subset=[ra_col, dec_col])
        if len(sp_df) == 0:
            st.warning("No hay filas con RA/Dec válidos después de la conversión.")
        else:
            ra_min, ra_max = float(sp_df[ra_col].min()), float(sp_df[ra_col].max())
            dec_min, dec_max = float(sp_df[dec_col].min()), float(sp_df[dec_col].max())

            tooltip_sp = ["kepid", ra_col, dec_col]
            if color_spatial in sp_df.columns:
                tooltip_sp.append(color_spatial)
            if size_spatial in sp_df.columns:
                tooltip_sp.append(size_spatial)
            if "prediction" in sp_df.columns:
                tooltip_sp.append("prediction")

            base_sp = alt.Chart(sp_df).mark_circle(opacity=0.9, size=80).encode(
                x=alt.X(f"{ra_col}:Q", title="Right Ascension (deg)"),
                y=alt.Y(f"{dec_col}:Q", title="Declination (deg)"),
                tooltip=[alt.Tooltip(f"{t}", format=".3f") if sp_df[t].dtype.kind in "fi" else alt.Tooltip(f"{t}") for t in tooltip_sp]
            )

            if color_spatial in sp_df.columns:
                sp_df[color_spatial] = sp_df[color_spatial].astype(str).fillna("nan")
                unique_vals = list(sp_df[color_spatial].dropna().unique())
                if len(unique_vals) == 2:
                    try:
                        sorted_vals = sorted(unique_vals, key=lambda v: float(v))
                    except Exception:
                        sorted_vals = sorted(unique_vals, key=lambda v: str(v))
                    domain = [str(v) for v in sorted_vals]
                    range_colors = ["#1f77b4", "#d62728"]
                    chart_sp = base_sp.encode(
                        color=alt.Color(f"{color_spatial}:N", title="Target",
                                        scale=alt.Scale(domain=domain, range=range_colors))
                    )
                else:
                    if sp_df[color_spatial].nunique() <= 12:
                        chart_sp = base_sp.encode(color=alt.Color(f"{color_spatial}:N", title="Target", legend=alt.Legend(orient="right")))
                    else:
                        chart_sp = base_sp.encode(color=alt.Color(f"{color_spatial}:Q", title="Target (numeric)", scale=alt.Scale(scheme="viridis")))
            else:
                chart_sp = base_sp

            zoom_x = [280, 304]
            zoom_y = [41, 49]
            x_ok = not (ra_max < zoom_x[0] or ra_min > zoom_x[1])
            y_ok = not (dec_max < zoom_y[0] or dec_min > zoom_y[1])

            if x_ok and y_ok:
                x_scale = alt.Scale(domain=zoom_x)
                y_scale = alt.Scale(domain=zoom_y)
            else:
                x_scale = alt.Scale(domain=[ra_min, ra_max])
                y_scale = alt.Scale(domain=[dec_min, dec_max])

            chart_sp = chart_sp.properties(width=900, height=600).encode(
                x=alt.X(f"{ra_col}:Q", scale=x_scale, title="Right Ascension (deg)"),
                y=alt.Y(f"{dec_col}:Q", scale=y_scale, title="Declination (deg)")
            ).interactive()

            st.altair_chart(chart_sp, use_container_width=True)

st.divider()

DATA_ORIG = Path(__file__).resolve().parent.parent.joinpath("Data", "1_cumulative_2025.csv")
st.markdown('<div class="section-header">Exploración del dataset original (KOI)</div>', unsafe_allow_html=True)
st.markdown(f"**Fuente:** `{DATA_ORIG}`")

if not DATA_ORIG.exists():
    st.warning("No se encontró ../Data/1_cumulative_2025.csv. Coloca el CSV y recarga la página.")
else:
    try:
        df_orig = pd.read_csv(DATA_ORIG, low_memory=False)
    except Exception as e:
        st.error(f"Error leyendo el CSV original: {e}")
        df_orig = None

    if df_orig is None or df_orig.empty:
        st.info("Dataset vacío o no se pudo leer.")
    else:
        st.markdown("**Clasificación de objetos Kepler (KOI)**")
        
        if "koi_disposition" in df_orig.columns:
            df_orig["koi_disposition"] = df_orig["koi_disposition"].astype(str).str.strip()
            counts = df_orig["koi_disposition"].value_counts(dropna=False)
            
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
            sns.countplot(
                data=df_orig,
                x="koi_disposition",
                order=counts.index,
                palette=colors[:len(counts)],
                ax=ax
            )
            
            ax.set_title("Clasificación de objetos Kepler (KOI)", fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel("Disposición", fontsize=12)
            ax.set_ylabel("Cantidad", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            
            total = len(df_orig)
            for p in ax.patches:
                height = p.get_height()
                if total > 0:
                    perc = f"{(height / total * 100):.1f}%"
                    ax.annotate(
                        perc,
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight='bold',
                        color="#2c3e50",
                        xytext=(0, 5),
                        textcoords="offset points"
                    )
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        st.write("")
        
        st.markdown("**Distribuciones de propiedades planetarias**")
        
        required_cols = ["koi_prad", "koi_period", "koi_teq"]
        available_cols = [col for col in required_cols if col in df_orig.columns]
        
        if len(available_cols) == 3:
            sns.set_style("whitegrid")
            plt.rcParams['font.size'] = 10
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            colors = ['#e74c3c', '#3498db', '#f39c12']
            
            valid_prad = df_orig["koi_prad"].dropna()
            if len(valid_prad) > 0:
                sns.histplot(
                    valid_prad, 
                    bins=50, 
                    log_scale=(True, False), 
                    ax=axes[0],
                    color=colors[0],
                    edgecolor='white',
                    alpha=0.8
                )
                axes[0].set_xlabel("Radio planetario [R⊕]", fontsize=11, fontweight='bold')
                axes[0].set_title("Distribución del radio planetario", fontsize=12, fontweight='bold', pad=10)
                axes[0].grid(True, alpha=0.3)
            
            valid_period = df_orig["koi_period"].dropna()
            if len(valid_period) > 0:
                sns.histplot(
                    valid_period, 
                    bins=50, 
                    log_scale=(True, False), 
                    ax=axes[1],
                    color=colors[1],
                    edgecolor='white',
                    alpha=0.8
                )
                axes[1].set_xlabel("Periodo orbital [días]", fontsize=11, fontweight='bold')
                axes[1].set_title("Distribución del periodo orbital", fontsize=12, fontweight='bold', pad=10)
                axes[1].grid(True, alpha=0.3)
            
            valid_teq = df_orig["koi_teq"].dropna()
            if len(valid_teq) > 0:
                sns.histplot(
                    valid_teq, 
                    bins=50, 
                    ax=axes[2],
                    color=colors[2],
                    edgecolor='white',
                    alpha=0.8
                )
                axes[2].set_xlabel("Temperatura de equilibrio [K]", fontsize=11, fontweight='bold')
                axes[2].set_title("Distribución de temperatura de equilibrio", fontsize=12, fontweight='bold', pad=10)
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(valid_prad) > 0:
                    st.metric("Radio mediano", f"{valid_prad.median():.2f} R⊕")
            with col2:
                if len(valid_period) > 0:
                    st.metric("Periodo mediano", f"{valid_period.median():.2f} días")
            with col3:
                if len(valid_teq) > 0:
                    st.metric("Temp. mediana", f"{valid_teq.median():.0f} K")
                    
        else:
            st.warning(f"No se encontraron todas las columnas necesarias. Disponibles: {available_cols}")
        
        st.markdown(
            """
            <div style='background-color: rgba(52, 152, 219, 0.1); padding: 15px; border-radius: 8px; margin-top: 20px;'>
            <p style='margin: 0; font-size: 14px;'>
            <strong>Nota:</strong> Los histogramas muestran las distribuciones de propiedades clave de los candidatos a exoplanetas. 
            El radio y periodo usan escala logarítmica en el eje X para visualizar mejor la amplia gama de valores.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

st.markdown('<div class="section-header">Sobre la interfaz</div>', unsafe_allow_html=True)
st.write(
    "Esta interfaz combina herramientas para transformar automáticamente CSVs KOI y para ejecutar "
    "modelos de clasificación entrenados con AutoGluon. En el panel de predicción puedes subir "
    "un CSV, aplicar las transformaciones necesarias y obtener predicciones en CSV; en el panel "
    "de entrenamiento puedes crear/derivar un target, ajustar parámetros de entrenamiento y "
    "reentrenar un predictor localmente."
)
st.info(
    "A la izquierda podrás encontrar links a estas páginas, además de un link a una página sobre nosotros y un canvas para desestresarte."
)

st.markdown(
    """
    <div style='margin-top: 60px; padding: 20px; text-align: center; font-size: 12px; color: #6c757d; border-top: 1px solid #e0e0e0;'>
    © Proyecto KOI Predictor – Visualización y herramientas para exploración de candidatos<br>
    Datos cargados desde <code>Back/predicciones.csv</code>
    </div>
    """, 
    unsafe_allow_html=True
)

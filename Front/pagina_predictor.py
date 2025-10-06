import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from transform_input import auto_transform, load_sample_stats

st.set_page_config(page_title="KOI Predictor - v0.0.1", layout="wide")
st.title("KOI Predictor")

def _load_predictor(path):
    """Callback: carga predictor y lo guarda en session_state"""
    try:
        p = Path(path).expanduser().resolve()
        predictor = TabularPredictor.load(p)
        st.session_state["predictor"] = predictor
        st.session_state["predictor_loaded"] = True
        st.session_state["predictor_error"] = ""
    except Exception as e:
        st.session_state["predictor_loaded"] = False
        st.session_state["predictor_error"] = str(e)

def _use_example():
    """Callback: carga test_input_real.csv desde disco a session_state['input_df']"""
    ex = Path("test_input_real.csv")
    if ex.exists():
        st.session_state["input_df"] = pd.read_csv(ex)
        st.session_state["uploader_name"] = str(ex)
        st.session_state["input_error"] = ""
    else:
        st.session_state["input_error"] = f"No existe {ex}"

def _upload_to_state():
    """Callback: lee el archivo subido y lo guarda en session_state"""
    # el UploadedFile está en st.session_state["uploaded_file_obj"] por la widget key
    uploaded = st.session_state.get("uploaded_file_obj")
    if uploaded is None:
        st.session_state["input_error"] = "No hay archivo subido."
        return
    try:
        df = pd.read_csv(uploaded)
        st.session_state["input_df"] = df
        st.session_state["uploader_name"] = getattr(uploaded, "name", "uploaded")
        st.session_state["input_error"] = ""
    except Exception as e:
        st.session_state["input_error"] = f"Error leyendo CSV: {e}"

def _transform_input():
    df = st.session_state.get("input_df")
    if df is None:
        st.session_state["input_error"] = "No hay input para transformar."
        return
    # intentar cargar sample stats (test_input_real.csv) si existe
    sample_path = Path("test_input_real.csv")
    sample_stats = load_sample_stats(sample_path) if sample_path.exists() else None
    transformed, info = auto_transform(df, sample_stats_df=sample_stats)
    st.session_state["input_df_transformed"] = transformed
    st.session_state["transform_info"] = info
    st.session_state["input_error"] = ""


def _predict():
    """Callback: corre la predicción y guarda el DataFrame resultante en session_state['pred_df']"""
    predictor = st.session_state.get("predictor")

    # Obtener preferentemente la versión transformada si existe y no es vacía
    transformed = st.session_state.get("input_df_transformed")
    original = st.session_state.get("input_df")

    if transformed is not None and isinstance(transformed, pd.DataFrame) and not transformed.empty:
        df = transformed
    elif original is not None and isinstance(original, pd.DataFrame) and not original.empty:
        df = original
    else:
        df = None

    if predictor is None:
        st.session_state["pred_error"] = "Predictor no cargado."
        return
    if df is None:
        st.session_state["pred_error"] = "No hay datos para predecir (input vacío o no cargado)."
        return

    try:
        preds = predictor.predict(df)
        # intentar probabilidades
        probs = None
        try:
            proba = predictor.predict_proba(df)
            if isinstance(proba, pd.DataFrame):
                if 1 in proba.columns:
                    probs = proba[1].values
                else:
                    probs = proba.iloc[:, -1].values
            else:
                arr = np.array(proba)
                if arr.ndim == 1:
                    probs = arr
                else:
                    probs = arr[:, -1]
        except Exception:
            probs = None

        out = df.copy()
        out["prediction"] = preds.values if hasattr(preds, "values") else preds
        if probs is not None:
            out["prob_positive"] = probs

        st.session_state["pred_df"] = out
        st.session_state["pred_error"] = ""
    except Exception as e:
        st.session_state["pred_error"] = f"Error en predict(): {e}"

# ---------- Inicializar session_state keys ----------
if "predictor_loaded" not in st.session_state:
    st.session_state["predictor_loaded"] = False
if "predictor_error" not in st.session_state:
    st.session_state["predictor_error"] = ""
if "input_df" not in st.session_state:
    st.session_state["input_df"] = None
if "uploader_name" not in st.session_state:
    st.session_state["uploader_name"] = ""
if "input_error" not in st.session_state:
    st.session_state["input_error"] = ""
if "pred_df" not in st.session_state:
    st.session_state["pred_df"] = None
if "pred_error" not in st.session_state:
    st.session_state["pred_error"] = ""

# ---------- Sidebar: predictor ----------
DEFAULT_PREDICTOR = os.getenv("PREDICTOR_PATH", "models/ag_predictor")
st.sidebar.header("Config")
predictor_path = st.sidebar.text_input("Ruta al predictor (carpeta)", value=DEFAULT_PREDICTOR)
st.sidebar.button("Cargar predictor", on_click=_load_predictor, args=(predictor_path,))

if st.session_state["predictor_loaded"]:
    st.sidebar.success("Predictor cargado ")
else:
    if st.session_state["predictor_error"]:
        st.sidebar.error("Error cargando predictor")
        st.sidebar.write(st.session_state["predictor_error"])

# ---------- Main: inputs ----------
st.markdown("## 1) Subir CSV con features (opcional)")
# file_uploader con clave y guardado temporal en session_state para permitir callback
uploaded = st.file_uploader("Sube CSV", type=["csv"], key="uploader_widget")

# Si el usuario acaba de seleccionar un archivo, guardamos el objeto UploadedFile en session_state y lanzamos callback
if uploaded is not None:
    st.session_state["uploaded_file_obj"] = uploaded
    # llama al callback que lee el archivo y lo pone como DataFrame en session_state
    _upload_to_state()

col1, col2 = st.columns([1,1])
with col1:
    st.button("Usar test_input_real.csv", on_click=_use_example)
with col2:
    st.write("Archivo actual:", st.session_state.get("uploader_name") or "Ninguno")

if st.session_state["input_error"]:
    st.error(st.session_state["input_error"])

# mostrar preview si hay df en session_state
if st.session_state.get("input_df") is not None:
    st.write("Vista previa (input):")
    st.dataframe(st.session_state["input_df"].head())
    st.button("Transformar para el modelo (crear features faltantes)", on_click=_transform_input, key="transform_btn")

# mostrar resultado de la transformación si existe
if st.session_state.get("input_df_transformed") is not None:
    st.info("Se aplicó la transformación automática. Revisa antes de predecir.")
    st.write("Vista previa (transformado):")
    st.dataframe(st.session_state["input_df_transformed"].head())
    info = st.session_state.get("transform_info", {})
    st.write(f"Columnas llenadas: {len(info.get('filled_cols', []))}; faltaban antes: {len(info.get('missing_before', []))}")

# ---------- Botón de predict (usando callback) ----------
st.markdown("### 2) Predecir")
st.button("Predecir", on_click=_predict, key="predict_button")

# mostrar error si ocurrió durante predict()
if st.session_state.get("pred_error"):
    st.error(st.session_state["pred_error"])

# mostrar resultados si existen
if st.session_state.get("pred_df") is not None:
    out = st.session_state["pred_df"]
    st.success(f"Predicciones generadas: {len(out)} filas")
    st.dataframe(out.head(200))
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar predicciones .csv", csv, file_name="preds_streamlit.csv", mime="text/csv")

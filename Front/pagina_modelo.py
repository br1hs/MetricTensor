# pagina_modelo.py
import os
from pathlib import Path
import time
import ast
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import json

st.set_page_config(page_title="KOI Trainer - v0.0.1", layout="wide")
st.title("KOI - Entrenar nuevo predictor (versión mínima)")


# ---------------- Helpers ----------------
def parse_drop_cols(text: str):
    if not text:
        return []
    # dividir por comas y limpiar
    return [c.strip() for c in text.split(",") if c.strip()]

def parse_hyperparams(text: str):
    # permite JSON o dict literal; devuelve None si vacío
    if not text or text.strip() == "":
        return None
    try:
        # usar ast.literal_eval para mayor seguridad
        return ast.literal_eval(text)
    except Exception:
        try:
            import json
            return json.loads(text)
        except Exception:
            raise ValueError("Hyperparameters: formato inválido. Usa JSON o dict literal.")

def parse_mapping_text(text: str):
    """
    Parsea texto tipo "CONFIRMED:1,FALSE POSITIVE:0,UNKNOWN:2" -> dict
    """
    if not text:
        return {}
    mapping = {}
    for part in text.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        mapping[k.strip()] = int(v.strip())
    return mapping

def apply_target_derivation(df: pd.DataFrame, source_col: str, mode: str, 
                            positive_value: str = None, mapping: dict | None = None,
                            new_target_name: str = "target", drop_source: bool = True):
    """
    Crea new_target_name en df:
     - mode == "binary_value": uses positive_value to set 1 else 0
     - mode == "mapping": uses mapping dict to map values -> ints; unknowns -> -1 (or fallback)
    Returns (df_modified, info_dict)
    """
    info = {"created": False, "errors": []}
    if source_col not in df.columns:
        info["errors"].append(f"Source column '{source_col}' not found.")
        return df, info

    series = df[source_col].astype(str).fillna("")

    if mode == "binary_value":
        if positive_value is None:
            info["errors"].append("positive_value requerido para modo binary_value.")
            return df, info
        df[new_target_name] = (series == positive_value).astype(int)
        info["created"] = True
        info["method"] = f"binary == '{positive_value}'"
    elif mode == "mapping":
        if not mapping:
            info["errors"].append("mapping vacío para modo mapping.")
            return df, info
        # aplicar mapping; valores no mapeados -> np.nan
        df[new_target_name] = series.map(mapping).astype("Int64")
        # fallback: rellenar NaNs con -1 (o podrías usar 0)
        df[new_target_name] = df[new_target_name].fillna(-1).astype(int)
        info["created"] = True
        info["method"] = f"mapping keys={list(mapping.keys())[:10]}"
    else:
        info["errors"].append(f"Modo desconocido: {mode}")
        return df, info

    if drop_source:
        try:
            df = df.drop(columns=[source_col])
            info["dropped_source"] = True
        except Exception as e:
            info["errors"].append(f"No se pudo dropear {source_col}: {e}")
            info["dropped_source"] = False

    return df, info

def default_save_path(base_dir="models", prefix="ag-retrained"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    p = Path(base_dir) / f"{prefix}-{ts}"
    return str(p)

# ---------------- UI ----------------
st.markdown("Sube un CSV con tus features y etiqueta (target). Luego selecciona la columna target, "
            "opcionalmente indica columnas a quitar (coma-separadas) y configura `time_limit` y `sample_size` para pruebas.")

uploaded = st.file_uploader("Sube CSV (para entrenar)", type=["csv"])
use_sample_from_disk = st.checkbox("Usar test_input_real.csv (si existe)", value=False)

# parámetros
drop_str = st.text_input("Columnas a quitar (coma-separadas)", value="")
sample_size = st.number_input("Sample size (0 = usar todo)", min_value=0, value=0, step=50)
time_limit = st.number_input("time_limit (segundos) para fit()", min_value=10, value=120, step=10)
presets = st.selectbox("Presets (calidad/tiempo)", options=["medium_quality", "good_quality", "best_quality"], index=0)
# hyperparams optional (we will NOT run HPO); user can pass simple hyperparameters dict
hyperparams_text = st.text_area("hyperparameters (opcional) — JSON o dict literal (ej. {'GBM':{}}) ", value="")
# options to reduce ensemble complexity
num_bag_folds = st.number_input("num_bag_folds (0 para desactivar bagging)", min_value=0, max_value=20, value=0, step=1)
# stacking: user asked "without the limit of stacking" — we will NOT set num_stack_levels so AutoGluon uses default.
# but we offer option to set to 0 to disable
num_stack_levels = st.number_input("num_stack_levels (-1 auto, 0 desactivar stacking, >=1 niveles)", min_value=-1, max_value=5, value=-1, step=1)

save_base = st.text_input("Carpeta base para guardar predictor (relativa)", value="models")
save_path_input = st.text_input("Ruta final (opcional) — deja vacío para auto", value="")
confirm_checkbox = st.checkbox("Confirmo que quiero iniciar entrenamiento en esta máquina (puede consumir CPU/RAM).", value=False)

# ---------------- Internal state / read data ----------------
if use_sample_from_disk:
    sample_path = Path("test_input_real.csv")
    if sample_path.exists():
        df_uploaded = pd.read_csv(sample_path)
        st.success("Cargando test_input_real.csv desde disco")
    else:
        df_uploaded = None
        st.warning("test_input_real.csv no encontrado en el directorio actual.")
else:
    df_uploaded = None
    if uploaded is not None:
        try:
            df_uploaded = pd.read_csv(uploaded)
            st.success(f"CSV cargado: filas={len(df_uploaded)}, columnas={len(df_uploaded.columns)}")
        except Exception as e:
            st.error(f"Error leyendo CSV: {e}")
            df_uploaded = None

# preview
if df_uploaded is not None:
    st.write("Preview del dataset:")
    st.dataframe(df_uploaded.head())

# target selection (dependiente del df)
target_col = None
if df_uploaded is not None:
    cols = df_uploaded.columns.tolist()
    target_col = st.selectbox("Selecciona la columna TARGET (label) para entrenar", options=cols)

st.markdown("### Opciones para derivar o transformar la columna target (si es texto)")

derive_mode = st.radio("¿Quieres derivar una nueva columna target desde una columna textual?", 
                      options=["No, ya tengo target numérico", "Sí — crear binario por valor", "Sí — usar mapping personalizado"],
                      index=0)

# valores que usamos más adelante; guardamos en session_state
st.session_state["derive_mode"] = derive_mode

if derive_mode != "No, ya tengo target numérico":
    # columna fuente (puede ser igual al target seleccionado)
    src_col = st.selectbox("Columna fuente (texto) para derivar target", options=df_uploaded.columns.tolist() if df_uploaded is not None else [])
    st.session_state["src_col_for_target"] = src_col

    if derive_mode == "Sí — crear binario por valor":
        pos_value = st.text_input("Valor POSITIVO (ej. CONFIRMED) — igual exacto al texto en la columna", value="CONFIRMED")
        st.session_state["pos_value"] = pos_value
    else:
        mapping_text = st.text_area("Mapping (ej. CONFIRMED:1,FALSE POSITIVE:0,NOT_DETECTED:0)", value="")
        st.session_state["mapping_text"] = mapping_text

    new_target_name = st.text_input("Nombre de la nueva columna target (ej. target)", value="target")
    st.session_state["new_target_name"] = new_target_name

    drop_source_checkbox = st.checkbox("Eliminar la columna fuente textual después de crear target?", value=True)
    st.session_state["drop_source_after_target"] = drop_source_checkbox
else:
    # si no derivar, pedir nombre de columna target existente
    existing_target = st.selectbox("Selecciona la columna target existente (ya usable)", options=df_uploaded.columns.tolist() if df_uploaded is not None else [])
    st.session_state["existing_target_name"] = existing_target

# ---------------- Training callback ----------------
def _train_button_callback():
    # validar confirmación
    if not st.session_state.get("confirm_checkbox"):
        st.session_state["train_status"] = "Cancelado por no confirmar checkbox."
        return

    df = st.session_state.get("df_uploaded")
    if df is None or df.empty:
        st.session_state["train_status"] = "No hay datos para entrenar."
        return

    target = st.session_state.get("selected_target")
    if not target:
        st.session_state["train_status"] = "No se seleccionó columna target."
        return

    # preparar df: dropear columnas solicitadas
    drop_cols = st.session_state.get("drop_cols_list", [])
    df2 = df.copy()

    derive_mode = st.session_state.get("derive_mode", "No, ya tengo target numérico")
    if derive_mode != "No, ya tengo target numérico":
        src_col = st.session_state.get("src_col_for_target")
        new_target_name = st.session_state.get("new_target_name", "target")
        drop_source = bool(st.session_state.get("drop_source_after_target", True))

        if derive_mode == "Sí — crear binario por valor":
            pos = st.session_state.get("pos_value", "CONFIRMED")
            df2, info_target = apply_target_derivation(df2, source_col=src_col, mode="binary_value",
                                                       positive_value=pos, new_target_name=new_target_name, drop_source=drop_source)
        else:
            mapping_text = st.session_state.get("mapping_text", "")
            mapping = parse_mapping_text(mapping_text)
            df2, info_target = apply_target_derivation(df2, source_col=src_col, mode="mapping",
                                                       mapping=mapping, new_target_name=new_target_name, drop_source=drop_source)
        # registrar info para mostrar al usuario
        st.session_state["last_target_info"] = info_target

        # ahora el target real que usaremos pasa a ser new_target_name
        target = new_target_name
    else:
        # no derivamos: usar el target seleccionado por el usuario (existing_target_name)
        target = st.session_state.get("selected_target")  # ya estabas guardando esto antes

    # --- Ahora filtramos drop_cols y aseguramos que target NO esté en la lista de drop ---
    drop_cols = st.session_state.get("drop_cols_list", [])
    if target in drop_cols:
        # quitarlo silenciosamente y notificar
        drop_cols = [c for c in drop_cols if c != target]
        st.session_state["train_status"] = (st.session_state.get("train_status","") + 
                                           f"\nAdvertencia: '{target}' estaba en la lista de columnas a eliminar y fue removido automáticamente.")

    cols_present_to_drop = [c for c in drop_cols if c in df2.columns]
    if cols_present_to_drop:
        df2 = df2.drop(columns=cols_present_to_drop)

    # sample
    sample_n = st.session_state.get("sample_size", 0)
    if sample_n and sample_n > 0 and sample_n < len(df2):
        df2 = df2.sample(sample_n, random_state=42).reset_index(drop=True)

    # separar train/val (holdout)
    try:
        train_df, val_df = train_test_split(df2, test_size=0.2, random_state=42, stratify=df2[target] if df2[target].nunique() > 1 else None)
    except Exception:
        # fallback sin stratify
        train_df, val_df = train_test_split(df2, test_size=0.2, random_state=42)

    # parse hyperparams
    try:
        hyper = parse_hyperparams(st.session_state.get("hyperparams_text", ""))
    except Exception as e:
        st.session_state["train_status"] = f"Hyperparameters parse error: {e}"
        return

    # construir save_path
    if st.session_state.get("save_path_input"):
        save_path = st.session_state.get("save_path_input")
    else:
        save_path = default_save_path(base_dir=st.session_state.get("save_base", "models"))

    # crear carpeta si no existe
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # iniciar entrenamiento
    st.session_state["train_status"] = f"Iniciando fit() -> guardando en {save_path} ..."
    try:
        start_t = time.time()
        predictor = TabularPredictor(label=target, path=save_path,eval_metric='roc_auc')
        fit_kwargs = dict(
            train_data=train_df,
            presets=st.session_state.get("presets", "medium_quality"),
            time_limit=int(st.session_state.get("time_limit", 120)),
            hyperparameters=hyper,
            num_bag_folds=int(st.session_state.get("num_bag_folds", 0)),
        )
        # num_stack_levels: if user set >=0, pass it; if -1, omit to use default
        nstack = int(st.session_state.get("num_stack_levels", -1))
        if nstack >= 0:
            fit_kwargs["num_stack_levels"] = nstack

        # NO hyperparameter_tune_kwargs (user requested no HPO)
        # Call fit
        with st.spinner("Entrenando... esto puede tardar. Revisa la terminal para logs detallados."):
            predictor = predictor.fit(**fit_kwargs, tuning_data=val_df)
        elapsed = time.time() - start_t
        st.session_state["train_status"] = f"Entrenamiento finalizado en {elapsed:.1f}s. Predictor guardado en: {save_path}"
        st.session_state["last_predictor_path"] = save_path
        st.session_state["last_predictor"] = predictor


        # ---------- CALCULAR FEATURE IMPORTANCE (Top 15) ----------
        try:
            if 'val_df' in locals() and val_df is not None and len(val_df) > 0:
                data_for_fi = val_df
            else:
                data_for_fi = train_df.sample(min(200, len(train_df)), random_state=42) if len(train_df) > 0 else None

            if data_for_fi is not None:
                # Intentamos varias firmas para compatibilidad con distintas versiones de AutoGluon
                fi = None
                last_err = None
                try:
                    fi = predictor.feature_importance(val_data=data_for_fi, silent=True)
                except Exception as e:
                    last_err = e
                    try:
                        fi = predictor.feature_importance(data=data_for_fi, silent=True)
                    except Exception as e2:
                        last_err = e2
                        try:
                            fi = predictor.feature_importance(data_for_fi, silent=True)
                        except Exception as e3:
                            last_err = e3
                            fi = None

                if fi is not None:
                    st.session_state["last_feature_importance"] = fi
                    st.markdown("#### Top 15 Feature Importance")
                    try:
                        st.dataframe(fi.head(15))
                        if "importance" in fi.columns:
                            top = fi.sort_values("importance", ascending=False).head(15)
                            st.bar_chart(top["importance"], sort="importance")
                    except Exception:
                        try:
                            fi_df = pd.DataFrame(fi)
                            st.dataframe(fi_df.head(15))
                        except Exception as e:
                            st.warning(f"No pude mostrar feature_importance: {e}")
                else:
                    st.warning(f"No se obtuvo feature_importance (último error: {last_err})")
            else:
                st.info("No hay datos suficientes para calcular feature importance.")
        except Exception as e:
            st.warning(f"No se pudo calcular feature_importance: {e}")
        # ----------------------------------------------------------------

        # mostrar leaderboard y evaluar en val set (si posible)
        try:
            lb = predictor.leaderboard(silent=True)
            # convertir a tabla pandas y guardar en session_state
            st.session_state["last_leaderboard"] = lb
        except Exception:
            st.session_state["last_leaderboard"] = None

        # evaluar en val set si existe
        try:
            y_true = val_df[target]
            y_pred = predictor.predict(val_df)
            from sklearn.metrics import accuracy_score, classification_report
            acc = accuracy_score(y_true, y_pred)
            st.session_state["last_eval"] = {"accuracy": float(acc), "n_val": len(val_df)}
        except Exception as e:
            st.session_state["last_eval"] = {"error": str(e)}
    except Exception as e:
        st.session_state["train_status"] = f"Error durante fit(): {e}"

# ---------------- Buttons / wiring ----------------
# botón de entrenamiento (habilitado solo cuando hay df y target, y confirm checkbox)
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.write("Carpeta destino:", save_path_input or default_save_path())
with col2:
    st.session_state["confirm_checkbox"] = confirm_checkbox
    st.session_state["save_base"] = save_base
    st.session_state["save_path_input"] = save_path_input.strip() or ""
    st.session_state["drop_cols_list"] = parse_drop_cols(drop_str)
    st.session_state["sample_size"] = int(sample_size)
    st.session_state["time_limit"] = int(time_limit)
    st.session_state["presets"] = presets
    st.session_state["hyperparams_text"] = hyperparams_text
    st.session_state["num_bag_folds"] = int(num_bag_folds)
    st.session_state["num_stack_levels"] = int(num_stack_levels)
    st.session_state["df_uploaded"] = df_uploaded
    st.session_state["selected_target"] = target_col

with col3:
    start_train = st.button("Iniciar entrenamiento (fit)", on_click=_train_button_callback)

# ---------------- Status / Results ----------------
st.markdown("### Estado / Resultados")
if st.session_state.get("train_status"):
    st.info(st.session_state.get("train_status"))

if st.session_state.get("last_leaderboard") is not None:
    st.markdown("LeaderBoard (modelos en el ensemble)")
    st.dataframe(st.session_state.get("last_leaderboard"))

if st.session_state.get("last_eval") is not None:
    st.markdown("Evaluación (holdout)")
    st.write(st.session_state.get("last_eval"))

if st.session_state.get("last_predictor_path"):
    st.success(f"Último predictor guardado en: {st.session_state.get('last_predictor_path')}")
    st.write("Puedes usarlo en la página de predicción con ruta relativa.")

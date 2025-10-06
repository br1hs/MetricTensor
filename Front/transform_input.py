from pathlib import Path
import numpy as np
import pandas as pd

predictor_features = ['kepid','koi_period','koi_time0bk','koi_time0','koi_eccen','koi_impact','koi_duration','koi_depth','koi_ror','koi_srho','koi_prad','koi_sma','koi_incl','koi_teq','koi_insol','koi_dor','koi_ldm_coeff2','koi_ldm_coeff1','koi_max_sngle_ev','koi_model_snr','koi_count','koi_num_transits','koi_tce_plnt_num','koi_steff','koi_slogg','koi_smet','koi_srad','koi_smass','ra','dec','koi_kepmag','koi_gmag','koi_rmag','koi_imag','koi_zmag','koi_jmag','koi_hmag','koi_kmag','koi_fwm_stat_sig','koi_fwm_sra','koi_fwm_sdec','koi_fwm_srao','koi_fwm_sdeco','koi_fwm_prao','koi_fwm_pdeco','koi_dicco_mra','koi_dicco_mdec','koi_dikco_mra','koi_dikco_mdec','kepler_ratio','temp_ratio','duration_anomaly','ultra_deep','log_period','log_depth','log_prad','log_teq','transit_ratio','temp_period_ratio','impact_squared','transit_morgan','caida_brillo','koi_fittype','koi_quarters','koi_sparprov','period_class']

drop_cols = [
    'koi_longp', 'koi_model_chisq', 'koi_model_dof', 'koi_ingress', 'koi_sage',
    'rowid', 'kepoi_name', 'kepler_name',
    'koi_vet_stat', 'koi_vet_date', 'koi_disp_prov', 'koi_parm_prov',
    'koi_tce_delivname', 'koi_datalink_dvr', 'koi_datalink_dvs',
    'koi_limbdark_mod', 'koi_trans_mod', 'koi_ldm_coeff3', 'koi_ldm_coeff4',
    'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_comment',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_max_mult_ev', 'koi_dicco_msky', 'koi_dikco_msky', 'koi_bin_oedp_sig',
]

def safe_log10(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = x > 0
    out[mask] = np.log10(x[mask])
    return out

def safe_sqrt(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = x >= 0
    out[mask] = np.sqrt(x[mask])
    return out

def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = (b != 0) & (~np.isnan(b))
    out[mask] = a[mask] / b[mask]
    return out

def classify_period_series(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, "unknown", dtype=object)
    mask_ultra = (p < 1)
    mask_short = (p >= 1) & (p < 10)
    mask_medium = (p >= 10) & (p < 100)
    mask_long = (p >= 100)
    out[mask_ultra] = "ultra_short"
    out[mask_short] = "short"
    out[mask_medium] = "medium"
    out[mask_long] = "long"
    return out

def load_sample_stats(sample_path: str | Path, nrows: int = 5000):
    """Carga un sample para calcular medianas de fallback (si existe)."""
    p = Path(sample_path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, nrows=nrows)
    except Exception:
        return None

def auto_transform(df_orig: pd.DataFrame, sample_stats_df: pd.DataFrame | None = None):
    """
    Transforma df_orig para contener las predictor_features.
    Devuelve (out_df, info) donde info = {'missing_before': [...], 'filled_cols': [...]}.
    """
    df = df_orig.copy()
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # estadisticas medianas
    if sample_stats_df is not None:
        numeric_cols = sample_stats_df.select_dtypes(include=[np.number]).columns.tolist()
        medians = sample_stats_df[numeric_cols].median(numeric_only=True).to_dict()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        medians = df[numeric_cols].median(numeric_only=True).to_dict()

    # cálculos seguros
    kp = pd.to_numeric(df.get('koi_period'), errors='coerce')
    srad = pd.to_numeric(df.get('koi_srad'), errors='coerce')
    sma = pd.to_numeric(df.get('koi_sma'), errors='coerce')
    prad = pd.to_numeric(df.get('koi_prad'), errors='coerce')
    impact = pd.to_numeric(df.get('koi_impact'), errors='coerce')
    duration = pd.to_numeric(df.get('koi_duration'), errors='coerce')
    steff = pd.to_numeric(df.get('koi_steff'), errors='coerce')
    srho = pd.to_numeric(df.get('koi_srho'), errors='coerce')
    teq = pd.to_numeric(df.get('koi_teq'), errors='coerce')
    depth = pd.to_numeric(df.get('koi_depth'), errors='coerce')
    ror = pd.to_numeric(df.get('koi_ror'), errors='coerce')

    with np.errstate(divide='ignore', invalid='ignore'):
        term_inside = ((1 + safe_div(prad, srad))**2) - (impact**2)
        term_inside = np.where(term_inside < 0, np.nan, term_inside)
        transit_morgan = safe_div(kp * srad, np.pi * sma) * safe_sqrt(term_inside)
    df['transit_morgan'] = transit_morgan

    with np.errstate(divide='ignore', invalid='ignore'):
        df['transit_ratio'] = safe_div(duration, (kp * 24))
    with np.errstate(divide='ignore', invalid='ignore'):
        df['temp_period_ratio'] = safe_div(steff, safe_sqrt(kp))
    df['impact_squared'] = impact**2
    df['kepler_ratio'] = srho * (kp**2)
    df['temp_ratio'] = safe_div(teq, steff)
    with np.errstate(divide='ignore', invalid='ignore'):
        expected_duration = 13 * srad * ((safe_div(kp, 365.25)) ** (1/3.0))
        df['duration_anomaly'] = (duration - expected_duration).abs()
    df['ultra_deep'] = (depth > 5000).astype(int).fillna(0).astype(int)
    df['log_period'] = safe_log10(kp)
    df['log_depth']  = safe_log10(depth)
    df['log_prad']   = safe_log10(pd.to_numeric(df.get('koi_prad'), errors='coerce'))
    df['log_teq']    = safe_log10(teq)
    df['caida_brillo'] = ror**2
    df['period_class'] = classify_period_series(kp)

    # categorías
    if 'koi_fittype' not in df.columns:
        df['koi_fittype'] = 'unknown'
    else:
        df['koi_fittype'] = df['koi_fittype'].fillna('unknown').astype(str)
    if 'koi_sparprov' not in df.columns:
        df['koi_sparprov'] = 'unknown'
    else:
        df['koi_sparprov'] = df['koi_sparprov'].fillna('unknown').astype(str)
    if 'koi_quarters' not in df.columns:
        df['koi_quarters'] = df.get('koi_quarters', np.nan)

    missing_before = [c for c in predictor_features if c not in df.columns]

    # rellenar faltantes
    filled = []
    for col in predictor_features:
        if col not in df.columns:
            if col in medians:
                df[col] = medians.get(col, 0.0)
            else:
                if col in ['koi_fittype', 'koi_sparprov', 'period_class']:
                    df[col] = 'unknown'
                else:
                    df[col] = 0.0
            filled.append(col)

    out_df = df[predictor_features].copy()
    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    for c in out_df.columns:
        if pd.api.types.is_numeric_dtype(out_df[c]):
            med = medians.get(c, 0.0)
            out_df[c] = out_df[c].fillna(med)
        else:
            out_df[c] = out_df[c].fillna('unknown')

    try:
        if 'kepid' in out_df.columns:
            out_df['kepid'] = out_df['kepid'].astype(int)
    except Exception:
        pass

    info = {"missing_before": missing_before, "filled_cols": filled}
    return out_df, info

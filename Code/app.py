from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Brasília Fire Risk", layout="wide")
st.title("🔥 Fire Risk – Brasília (daily)")

# ===== 1) carregar modelo
@st.cache_resource
def load_model():
    # tenta várias localizações seguras
    candidates = [
        Path(__file__).parent / "modelo_brasilia.pkl",          # mesma pasta do app.py
        Path.cwd() / "modelo_brasilia.pkl",                      # pasta onde corriste o streamlit
        Path(r"C:\Users\sofia\Documents\Data_Analytics_Ironhack\Projects\google_brain\Code\modelo_brasilia.pkl"),
    ]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    st.error(
        "❌ Não encontrei `modelo_brasilia.pkl`.\n\n"
        "Procurei em:\n- " + "\n- ".join(str(p) for p in candidates) +
        "\n\nMove o ficheiro para a mesma pasta do `app.py` ou ajusta o caminho."
    )
    st.stop()

model = load_model()

# mesmas features usadas no deploy (sem focos_*):
FEATURES = [
    "precipitacao_mm","radiacao_media","temp_media","temp_max","temp_min",
    "humidade_media","vento_rajada_max","vento_vel_media",
    "chuva_7d","temp_7d","humidade_7d","mes","dia_semana"
]

st.markdown("Carrega um CSV diário com as colunas base: "
            "`Data, precipitacao_mm, radiacao_media, temp_media, temp_max, temp_min, humidade_media, vento_rajada_max, vento_vel_media`.")

file = st.file_uploader("Upload CSV (INMET diário)", type=["csv"])

def build_features(df):
    df = df.copy()
    # Data
    date_col = "Data" if "Data" in df.columns else "data"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    # rolling (shift=1 para evitar leakage)
    df["chuva_7d"]   = df["precipitacao_mm"].rolling(7, min_periods=1).sum().shift(1)
    df["temp_7d"]    = df["temp_media"].rolling(7, min_periods=1).mean().shift(1)
    df["umidade_7d"] = df["humidade_media"].rolling(7, min_periods=1).mean().shift(1)

    df["mes"] = df[date_col].dt.month
    df["dia_semana"] = df[date_col].dt.weekday

    # garantir todas as FEATURES (se alguma faltar, preenche com 0)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0

    # remove primeiras linhas com NaN por causa do shift
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df, date_col

if file is not None:
    raw = pd.read_csv(file)
    df, date_col = build_features(raw)

    if df.empty:
        st.warning("Depois do cálculo de rolling/shift não sobraram linhas. Carrega mais dias (>= 8).")
        st.stop()

    # ===== 2) predições
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred  = model.predict(df[FEATURES])

    out = df[[date_col]].copy()
    out.rename(columns={date_col: "Data"}, inplace=True)
    out["fire_probability"] = proba
    # threshold ajustável
    th = st.slider("Decision threshold", 0.1, 0.9, 0.50, 0.01)
    out["fire_prediction"] = (out["fire_probability"] >= th).astype(int)

    st.subheader("Resultados")
    st.dataframe(out.tail(20), use_container_width=True)

    # ===== 3) gráfico
    out_plot = out.copy()
    out_plot["prob_7d"] = out_plot["fire_probability"].rolling(7, min_periods=1).mean()
    st.line_chart(
        out_plot.set_index("Data")[["fire_probability","prob_7d"]],
        height=300
    )
    st.caption("Linha sólida = probabilidade diária; linha suavizada = média móvel 7 dias.")

    # ===== 4) métricas simples (sem target futuro, mostramos apenas distribuição)
    st.write(f"**Dias sinalizados com threshold {th:.2f}:** {int(out['fire_prediction'].sum())} de {len(out)}")

    # ===== 5) download
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="predicoes_streamlit.csv", mime="text/csv")
else:
    st.info("👆 Carrega um CSV para gerar as previsões.")

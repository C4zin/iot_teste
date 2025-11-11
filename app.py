import streamlit as st
import tempfile
import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
import plotly.express as px
import time
import base64

# ---------------- CONFIGURAÇÃO GERAL ----------------
st.set_page_config(
    page_title="SafeWork - Projeto Global Solution",
    page_icon="🦺",
    layout="wide"
)

# ---------------- CSS MODERNO ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f0f0f;
    color: #e0e0e0;
}
.block-container {
    padding: 2rem 3rem;
    background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
    border-radius: 12px;
    box-shadow: 0px 0px 15px rgba(255, 0, 0, 0.15);
}
h1, h2, h3 {
    color: #b22222 !important;
    font-weight: 800 !important;
    text-transform: uppercase;
}
.stSidebar {
    background-color: #1a1a1a;
}
div[data-testid="stFileUploader"] {
    background-color: #1c1c1c !important;
    border: 2px solid #b22222 !important;
    border-radius: 10px !important;
}
div[data-testid="stDownloadButton"] button {
    background-color: #b22222;
    color: #fff;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    border: none;
    transition: all 0.3s ease-in-out;
}
div[data-testid="stDownloadButton"] button:hover {
    background-color: #e63946;
    transform: scale(1.05);
}
.stProgress > div > div > div {
    background-color: #b22222;
}
div[data-testid="stMetricValue"] {
    color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CABEÇALHO ----------------
logo_path = "logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px;">
            <img src="data:image/png;base64,{logo_base64}" width="45"/>
            <h1>SafeWork - Projeto Global Solution</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("🦺 SafeWork - Projeto Global Solution")

st.markdown("### Análise de Trabalhadores")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configurações")
confidence = st.sidebar.slider("Nível de confiança da detecção", 0.1, 1.0, 0.35, 0.05)
st.sidebar.markdown("---")
st.sidebar.info("📹 Faça upload de um vídeo para analisar o uso de EPIs.")

uploaded_file = st.sidebar.file_uploader("Enviar vídeo (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

# ---------------- CARREGAR MODELO ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- FUNÇÃO PRINCIPAL ----------------
def analyze_video(input_path, output_path, conf):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    progress = st.progress(0)
    frame_metrics = []
    counters = {"person": 0, "helmet": 0, "vest": 0, "mask": 0, "unknown": 0}

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        names = results[0].names
        frame_data = {"frame": i, "person": 0, "helmet": 0, "vest": 0, "mask": 0}
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = names[cls].lower()
            if "person" in label:
                counters["person"] += 1
                frame_data["person"] += 1
            elif any(k in label for k in ["helmet", "hardhat"]):
                counters["helmet"] += 1
                frame_data["helmet"] += 1
            elif "vest" in label:
                counters["vest"] += 1
                frame_data["vest"] += 1
            elif "mask" in label:
                counters["mask"] += 1
                frame_data["mask"] += 1
            else:
                counters["unknown"] += 1

        frame_metrics.append(frame_data)
        out.write(annotated)
        progress.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    progress.empty()
    return counters, pd.DataFrame(frame_metrics)

# ---------------- EXECUÇÃO ----------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    output_path = os.path.join(tempfile.gettempdir(), "video_analisado.mp4")
    with st.spinner("🔍 Analisando vídeo, isso pode levar alguns minutos..."):
        counters, frame_data = analyze_video(temp_input_path, output_path, confidence)

    if counters:
        st.balloons()
        st.success("✅ Análise concluída com sucesso!")

        st.markdown("## 📊 Dashboard de Detecção de EPI")

        total_epi = counters["helmet"] + counters["vest"] + counters["mask"]
        conformidade = (total_epi / counters["person"] * 100) if counters["person"] > 0 else 0

        # Painel de conformidade dinâmico
        if conformidade >= 90:
            color = "#2ecc71"
        elif conformidade >= 70:
            color = "#f1c40f"
        else:
            color = "#e74c3c"

        st.markdown(f"""
        <div style='padding:1rem;border-radius:10px;background-color:{color};text-align:center;margin-top:10px;margin-bottom:20px;'>
        <h3 style='color:white;'>Conformidade Estimada: {conformidade:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pessoas", counters["person"])
        col2.metric("Capacetes", counters["helmet"])
        col3.metric("Coletes", counters["vest"])
        col4.metric("Máscaras", counters["mask"])

        # -------- GRÁFICOS --------
        data = pd.DataFrame({
            "Tipo": ["Pessoas", "Capacetes", "Coletes", "Máscaras", "Outros"],
            "Quantidade": [
                counters["person"], counters["helmet"], counters["vest"],
                counters["mask"], counters["unknown"]
            ]
        })

        fig_bar = px.bar(
            data,
            x="Tipo",
            y="Quantidade",
            color="Tipo",
            text="Quantidade",
            title="Distribuição de Detecções por Tipo",
            color_discrete_sequence=["#b22222", "#8b0000", "#696969", "#a9a9a9", "#2f2f2f"]
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Gráfico temporal
        if not frame_data.empty:
            melted = frame_data.melt(id_vars="frame", var_name="Categoria", value_name="Contagem")
            fig_line = px.line(
                melted, x="frame", y="Contagem", color="Categoria",
                title="Evolução das Detecções ao Longo do Vídeo",
                color_discrete_sequence=["#b22222", "#8b0000", "#696969", "#a9a9a9"]
            )
            st.plotly_chart(fig_line, use_container_width=True)

        fig_pie = px.pie(
            data,
            names="Tipo",
            values="Quantidade",
            title="Proporção de Detecções por Tipo",
            color_discrete_sequence=["#b22222", "#8b0000", "#696969", "#a9a9a9", "#2f2f2f"]
        )
        fig_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_pie, use_container_width=True)

        # -------- DOWNLOAD DO VÍDEO --------
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Baixar vídeo analisado",
                data=f,
                file_name="video_analisado.mp4",
                mime="video/mp4"
            )

        os.remove(temp_input_path)

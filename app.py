import streamlit as st
import tempfile
import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
import plotly.express as px

# ---------------- CONFIGURAÇÃO ----------------
st.set_page_config(page_title="Analisador de EPI", page_icon="🦺", layout="wide")

# ---------------- CSS CUSTOM ----------------
st.markdown("""
    <style>
    /* Fundo geral */
    .stApp {
        background-color: #0f0f0f;
        color: #e0e0e0;
    }

    /* Caixa principal */
    .block-container {
        padding: 2rem 3rem;
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-radius: 12px;
        box-shadow: 0px 0px 15px rgba(255, 0, 0, 0.15);
    }

    /* Cabeçalhos */
    h1, h2, h3 {
        color: #b22222 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }

    /* Botões */
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

    /* Métricas */
    div[data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
    }

    /* Upload box */
    div[data-testid="stFileUploader"] {
        background-color: #1c1c1c !important;
        border: 2px dashed #b22222 !important;
        border-radius: 10px !important;
        color: #d3d3d3 !important;
    }

    /* Mensagens (info/sucesso/erro) */
    .stAlert {
        border-radius: 8px;
    }
    .stAlert div {
        color: #fff !important;
    }

    .stAlert[data-baseweb="alert"]:has(div[data-testid="stNotificationContentSuccess"]) {
        background-color: rgba(178, 34, 34, 0.2) !important;
        border-left: 5px solid #b22222 !important;
    }

    /* Tabelas e gráficos */
    .plotly {
        background-color: transparent !important;
    }

    /* Barra de progresso */
    .stProgress > div > div > div {
        background-color: #b22222;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TÍTULO ----------------
st.title("🦺 Analisador de Vídeo de EPI com IA")
st.markdown("### Utilize inteligência artificial para verificar automaticamente o uso de EPIs em vídeos.")
st.markdown("---")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Envie um vídeo (.mp4, .mov, .avi)", 
    type=["mp4", "mov", "avi"]
)

# ---------------- MODELO ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- FUNÇÃO DE ANÁLISE ----------------
def analyze_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress = st.progress(0)

    counters = {"person": 0, "helmet": 0, "vest": 0, "mask": 0, "unknown": 0}

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.35, verbose=False)
        annotated = results[0].plot()

        names = results[0].names
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = names[cls].lower()
            if "person" in label:
                counters["person"] += 1
            elif any(k in label for k in ["helmet", "hardhat"]):
                counters["helmet"] += 1
            elif "vest" in label:
                counters["vest"] += 1
            elif "mask" in label:
                counters["mask"] += 1
            else:
                counters["unknown"] += 1

        out.write(annotated)
        progress.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    progress.empty()
    return counters

# ---------------- EXECUÇÃO ----------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    output_path = os.path.join(tempfile.gettempdir(), "video_analisado.mp4")
    st.info("🔍 Analisando vídeo — isso pode levar alguns minutos...")

    counters = analyze_video(temp_input_path, output_path)

    if counters:
        st.success("✅ Análise concluída com sucesso!")
        st.markdown("## 📊 Dashboard de Detecção de EPI")

        total_epi = counters["helmet"] + counters["vest"] + counters["mask"]
        conformidade = (total_epi / counters["person"] * 100) if counters["person"] > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pessoas detectadas", counters["person"])
        col2.metric("Capacetes", counters["helmet"])
        col3.metric("Coletes", counters["vest"])
        col4.metric("Máscaras", counters["mask"])

        st.markdown(f"### 🧮 Conformidade estimada: **{conformidade:.1f}%**")

        # -------- GRÁFICOS --------
        data = pd.DataFrame({
            "Tipo": ["Pessoas", "Capacetes", "Coletes", "Máscaras", "Outros"],
            "Quantidade": [
                counters["person"],
                counters["helmet"],
                counters["vest"],
                counters["mask"],
                counters["unknown"]
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

        fig_pie = px.pie(
            data,
            names="Tipo",
            values="Quantidade",
            title="Proporção de Detecções por Tipo",
            color_discrete_sequence=["#b22222", "#8b0000", "#696969", "#a9a9a9", "#2f2f2f"]
        )
        fig_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_pie, use_container_width=True)

        # -------- BOTÃO DE DOWNLOAD --------
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Baixar vídeo analisado",
                data=f,
                file_name="video_analisado.mp4",
                mime="video/mp4"
            )

        os.remove(temp_input_path)

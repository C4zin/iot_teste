import streamlit as st
import tempfile
import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
import plotly.express as px

# ---------------- CONFIGURAÇÃO ----------------
st.set_page_config(
    page_title="Analisador de EPI (PPE)",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 Analisador de Vídeo de EPI com IA")
st.markdown("Envie um vídeo para análise e visualize um **dashboard interativo** com as estatísticas de detecção de EPIs.")

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

    # Contadores globais
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

        # ---------------- DASHBOARD GRÁFICO ----------------
        st.subheader("📊 Dashboard de Detecção de EPI")

        total_epi = counters["helmet"] + counters["vest"] + counters["mask"]
        conformidade = (total_epi / counters["person"] * 100) if counters["person"] > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pessoas detectadas", counters["person"])
        col2.metric("Capacetes", counters["helmet"])
        col3.metric("Coletes", counters["vest"])
        col4.metric("Máscaras", counters["mask"])

        # Gráfico de barras
        data = pd.DataFrame({
            "Tipo": ["Pessoas", "Capacetes", "Coletes", "Máscaras", "Desconhecido"],
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
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Gráfico de pizza
        fig_pie = px.pie(
            data,
            names="Tipo",
            values="Quantidade",
            title="Proporção de Detecções por Tipo",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(f"### 🧮 Conformidade estimada: **{conformidade:.1f}%**")

        # ---------------- BOTÃO DE DOWNLOAD ----------------
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Baixar vídeo analisado",
                data=f,
                file_name="video_analisado.mp4",
                mime="video/mp4"
            )

        os.remove(temp_input_path)

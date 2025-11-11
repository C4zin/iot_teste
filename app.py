import streamlit as st
import tempfile
import cv2
import os
import numpy as np
from ultralytics import YOLO
import pandas as pd

# ---------------- CONFIGURAÇÃO ----------------
st.set_page_config(page_title="Analisador de EPI (PPE)", page_icon="🦺", layout="wide")
st.title("🦺 Analisador de Vídeo de EPI com IA")
st.markdown("Envie um vídeo e visualize as estatísticas de detecção de EPIs (capacete, colete, máscara, pessoas).")

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

        # Conta classes detectadas
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

        # ---------------- DASHBOARD ----------------
        st.subheader("📊 Estatísticas de Detecção")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pessoas detectadas", counters["person"])
        col2.metric("Capacetes", counters["helmet"])
        col3.metric("Coletes", counters["vest"])
        col4.metric("Máscaras", counters["mask"])

        total_epi = counters["helmet"] + counters["vest"] + counters["mask"]
        conformidade = (
            (total_epi / counters["person"] * 100) if counters["person"] > 0 else 0
        )

        st.markdown(f"### ✅ Conformidade estimada de EPI: **{conformidade:.1f}%**")

        # Tabela detalhada
        df = pd.DataFrame.from_dict(counters, orient="index", columns=["Quantidade"])
        st.dataframe(df.style.format("{:.0f}"))

        # ---------------- BOTÃO DE DOWNLOAD ----------------
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Baixar vídeo analisado",
                data=f,
                file_name="video_analisado.mp4",
                mime="video/mp4"
            )

        # Remove arquivo temporário de entrada
        os.remove(temp_input_path)

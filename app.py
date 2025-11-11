import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import numpy as np
from PIL import Image
import time

st.set_page_config(
    page_title="Análise de EPI com IA",
    layout="wide",
    page_icon="🦺"
)

st.title("🦺 Sistema de Análise de EPI com IA")
st.markdown(
    "Envie um vídeo para o sistema detectar automaticamente **pessoas com e sem EPI** (capacete, colete, luvas etc.)."
)

# --- Upload de vídeo ---
uploaded_file = st.file_uploader("Envie um vídeo para análise (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

# --- Carregar modelo YOLO ---
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # pode ser substituído por um modelo customizado
    return model

model = load_model()

# --- Função de processamento ---
def analyze_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    frame_placeholder = st.empty()

    detections_summary = {"with_epi": 0, "without_epi": 0}

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Heurística simples: se detectar pessoa sem capacete/colete → sem EPI
        names = results[0].names
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = names[cls].lower()
            if "person" in label:
                detections_summary["without_epi"] += 1
            elif any(epi in label for epi in ["helmet", "vest", "glove"]):
                detections_summary["with_epi"] += 1

        out.write(annotated_frame)

        # Atualizar UI
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        progress_bar.progress((i + 1) / total_frames)

    cap.release()
    out.release()
    return detections_summary

# --- Execução ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    st.info("🔍 Analisando vídeo, isso pode levar alguns minutos...")

    output_path = os.path.join(tempfile.gettempdir(), "output_analise.mp4")
    summary = analyze_video(input_path, output_path)

    # Mostrar resultado
    st.success("✅ Análise concluída!")
    st.video(output_path)

    st.subheader("📊 Relatório de Detecção")
    total = summary["with_epi"] + summary["without_epi"]
    epi_percent = (summary["with_epi"] / total * 100) if total > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("Com EPI", summary["with_epi"])
    col2.metric("Sem EPI", summary["without_epi"])

    st.markdown(f"**Conformidade estimada:** {epi_percent:.1f}%")

else:
    st.info("⬆️ Envie um vídeo para iniciar a análise.")

import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Análise de Vídeo PPE", layout="wide")
st.title("🎥 Analisador de Vídeos - Equipamentos de Proteção (PPE)")

model = YOLO("yolov8n.pt")  # Modelo leve

def analyze_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo. Verifique o arquivo enviado.")
        return "Erro ao abrir o vídeo"

    frame_placeholder = st.empty()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    # Codificador para salvar vídeo processado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # Conversão para exibir no Streamlit
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Evita erro caso frame_rgb venha vazio
        if frame_rgb is not None and frame_rgb.size > 0:
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        out.write(annotated_frame)

        processed_frames += 1
        progress_bar.progress(min(processed_frames / frame_count, 1.0))

    cap.release()
    out.release()
    st.success("✅ Análise concluída!")
    st.video(output_path)
    return "Análise finalizada com sucesso"

uploaded_file = st.file_uploader(
    "Envie um vídeo para análise (.mp4, .mov, .avi)", 
    type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        temp_output_path = temp_output.name

    st.info("🔍 Analisando vídeo, isso pode levar alguns minutos...")
    analyze_video(temp_input_path, temp_output_path)

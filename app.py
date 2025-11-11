import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="Analisador de Vídeo PPE", page_icon="🎥", layout="centered")

st.title("🎥 Analisador de Vídeo de EPI/PPE")
st.write("Envie um vídeo para análise (.mp4, .mov, .avi)")

uploaded_file = st.file_uploader(
    "Arraste ou selecione um arquivo de vídeo", 
    type=["mp4", "mov", "avi", "mpeg"]
)

model = YOLO("yolov8n.pt")  # Modelo leve e compatível com Streamlit Cloud

def analyze_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Erro ao abrir o vídeo. Verifique o arquivo enviado.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_placeholder = st.empty()
    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        out.write(annotated)
        progress.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    progress.empty()
    frame_placeholder.empty()

    st.success("✅ Análise concluída com sucesso!")

    # Exibe apenas o botão de download (sem preview)
    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Baixar vídeo analisado",
            data=f,
            file_name="video_analisado.mp4",
            mime="video/mp4"
        )

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    temp_output_path = os.path.join(tempfile.gettempdir(), "output_analisado.mp4")

    st.info("🔍 Analisando vídeo, isso pode levar alguns minutos...")
    analyze_video(temp_input_path, temp_output_path)

    os.remove(temp_input_path)

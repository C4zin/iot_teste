import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Configuração da página
# -------------------------------
st.set_page_config(page_title="Analisador de Vídeos PPE", page_icon="🦺", layout="wide")

st.title("🦺 Analisador de Vídeos PPE")
st.write("Envie um vídeo para análise (.mp4, .mov, .avi)")

# -------------------------------
# Upload do vídeo
# -------------------------------
uploaded_file = st.file_uploader("Drag and drop file here", type=["mp4", "mov", "avi", "mpeg"])

# Cria o placeholder de status
status_placeholder = st.empty()
frame_placeholder = st.empty()
progress_bar = st.progress(0)

# -------------------------------
# Função principal de análise
# -------------------------------
def analyze_video(input_path, output_path):
    model = YOLO("yolov8n.pt")  # modelo leve (substitua se quiser outro)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Erro ao abrir o vídeo.")
        return "Erro: vídeo não pôde ser aberto."

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configura o vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # fim do vídeo

        # Garante que o frame é válido
        if frame is None or frame.size == 0:
            st.warning("Frame inválido detectado, pulando...")
            continue

        # Executa a predição
        results = model(frame, verbose=False)

        # Renderiza resultados
        annotated_frame = results[0].plot()

        # Atualiza visualização no app
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Escreve no vídeo final
        out.write(annotated_frame)

        # Atualiza progresso
        processed += 1
        progress = int((processed / frame_count) * 100)
        progress_bar.progress(min(progress, 100))

    cap.release()
    out.release()

    return f"✅ Análise concluída! {processed} frames processados."

# -------------------------------
# Execução principal
# -------------------------------
if uploaded_file is not None:
    # Salva o vídeo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = os.path.join(tempfile.gettempdir(), "output.mp4")

    st.info("🔍 Analisando vídeo, isso pode levar alguns minutos...")

    summary = analyze_video(input_path, output_path)

    # Exibe mensagem final
    status_placeholder.success(summary)

    # Disponibiliza download
    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Baixar vídeo analisado",
            data=f,
            file_name="video_analisado.mp4",
            mime="video/mp4"
        )

else:
    st.warning("Envie um vídeo para iniciar a análise.")

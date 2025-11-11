import os
import subprocess
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="EPITrack - Dashboard", layout="wide")

st.title("EPITrack — Análise de vídeos e Dashboard de EPIs")
st.markdown("Envie um vídeo de trabalhadores; o sistema processará e retornará `Analisado.mp4` com anotações e um CSV com as métricas.")

uploaded = st.file_uploader("Envie um vídeo (mp4, avi, mkv, mov)", type=["mp4","avi","mkv","mov"])
model_choice = st.selectbox("Modelo", ["Auto-download (Construction-PPE)", "Envie seu .pt"])

custom_model = None
if model_choice.endswith("Envie seu .pt"):
    uploaded_model = st.file_uploader("Envie pesos (.pt)", type=["pt"])
    if uploaded_model is not None:
        model_path = os.path.join("weights_cache", uploaded_model.name)
        os.makedirs("weights_cache", exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(uploaded_model.read())
        custom_model = model_path

conf = st.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
overlap = st.slider("Overlap mínimo p/ considerar EPI (%)", 1, 90, 10, 1)
maxframes = st.number_input("Limite de frames (0 = todos)", min_value=0, value=0, step=1)

if st.button("Processar vídeo") and uploaded is not None:
    in_path = os.path.join("uploads", uploaded.name)
    os.makedirs("uploads", exist_ok=True)
    with open(in_path, "wb") as f:
        f.write(uploaded.read())
    out_video = "Analisado.mp4"
    out_csv = "results_epi.csv"
    model_arg = custom_model if custom_model else "yolov8n_epi.pt"
    # Run the detection script (assumes python detect_epi_video.py exists in same folder)
    cmd = [
        "python", "detect_epi_video.py",
        "--input", in_path,
        "--output", out_video,
        "--csv", out_csv,
        "--model", model_arg,
        "--conf", str(conf),
        "--overlap", str(overlap/100.0),
    ]
    if maxframes>0:
        cmd += ["--maxframes", str(int(maxframes))]

    with st.spinner("Processando vídeo — isto pode demorar dependendo do tamanho e da GPU/CPU..."):
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            st.error(f"Falha no processamento: {e}")
            st.stop()

    st.success("Processamento concluído!")

    # show video and metrics
    if os.path.exists(out_video):
        st.video(out_video)

    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        st.write("### Métricas por frame (amostra)")
        st.dataframe(df.head(100))
        st.download_button("Baixar CSV", df.to_csv(index=False).encode("utf-8"), file_name="results_epi.csv", mime="text/csv")

    # simples dashboard
    if os.path.exists(out_csv):
        import matplotlib.pyplot as plt
        df = pd.read_csv(out_csv)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Frames processados", int(df['frame'].max()))
            st.metric("Pessoas únicas (máx)", int(df['people_unicas'].max()))
        with col2:
            st.metric("Pico pessoas/frame", int(df['people_no_frame'].max()))
            st.metric("Pessoas sem EPI (total)", int(df['people_without_epi'].sum()))

        st.write("### Séries temporais")
        st.line_chart(df.set_index('frame')[['people_no_frame','people_with_epi','people_without_epi']])
else:
    st.info("Envie um vídeo acima e clique em 'Processar vídeo' para começar.")

# streamlit_app.py
import os
import time
import sys
import glob
import pickle
import shutil
import importlib
import subprocess
import numpy as np
import streamlit as st

# ============================================================
# 🔧 Garantia de dependências
# ============================================================
def ensure(pkg, pip_name=None):
    pip_name = pip_name or pkg
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

def ensure_cv2_headless():
    try:
        import cv2  # noqa
        if hasattr(cv2, "__file__") and "headless" not in (cv2.__file__ or "").lower():
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y",
                                   "opencv-python", "opencv-contrib-python"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "opencv-python-headless==4.10.0.84"])
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "opencv-python-headless==4.10.0.84"])

ensure("ultralytics")
ensure("supervision", "supervision==0.21.0")
ensure_cv2_headless()
ensure("lapx", "lapx>=0.5.9")
ensure("numpy", "numpy<2")

import cv2  # noqa
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import supervision as sv

# ============================================================
# Cache dedicado
# ============================================================
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", ".ultra_cache")
os.makedirs(os.environ["ULTRALYTICS_CACHE_DIR"], exist_ok=True)

def _clean_ultralytics_cache_for(weights_name: str):
    try:
        stem = os.path.splitext(os.path.basename(weights_name))[0]
        candidates = []
        weights_dir = SETTINGS.get("weights_dir", None)
        if weights_dir and os.path.isdir(weights_dir):
            candidates += glob.glob(os.path.join(weights_dir, f"{stem}*"))
        ultra_cache = os.environ.get("ULTRALYTICS_CACHE_DIR")
        if ultra_cache and os.path.isdir(ultra_cache):
            candidates += glob.glob(os.path.join(ultra_cache, "*"))
        torch_home = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
        if torch_home and os.path.isdir(torch_home):
            candidates += glob.glob(os.path.join(torch_home, "**", f"*{stem}*"), recursive=True)
        for p in set(candidates):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass

# ============================================================
# Streamlit + CSS
# ============================================================
st.set_page_config(page_title="Pessoas + PPE Track", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e0e0e; color: #f5f5f5; font-family: 'Segoe UI', Roboto, sans-serif; }
    h1, h2, h3, h4 { color: #d62828 !important; font-weight: 700 !important; letter-spacing: 0.5px; }
    section[data-testid="stSidebar"] { background-color: #161616 !important; border-right: 2px solid #d62828; }
    .stButton>button { background-color: #d62828 !important; color: white !important;
        border: none; border-radius: 6px; font-weight: 600; padding: 0.6em 1.2em; transition: 0.3s; }
    .stButton>button:hover { background-color: #e94c4c !important; transform: scale(1.05); }
    .stFileUploader { margin-bottom: 10px !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #1a1a1a !important; border: 2px dashed #d62828 !important;
        color: #ccc !important; border-radius: 8px; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1a1a1a !important; color: #fff !important;
        border: 1px solid #d62828 !important; border-radius: 4px !important; }
    .stSlider>div>div>div>div { background: linear-gradient(to right, #d62828, #8b0000); }
    [data-testid="stMetricValue"] { color: #d62828 !important; font-weight: 700 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "rt_logs" not in st.session_state:
    st.session_state["rt_logs"] = []

st.title("🧑‍🏭 PeopleTrack + PPE — YOLOv8 + ByteTrack")
st.caption("Detecção e rastreamento de pessoas com EPIs (capacete, colete, etc.)")

tab_sys, tab_app, tab_dash = st.tabs(["🖥️ Sistema", "🎬 Processamento", "📊 Dashboard"])

# ============================================================
# Aba Sistema
# ============================================================
with tab_sys:
    import platform
    st.subheader("Informações do Sistema")
    st.write({
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "ultralytics": __import__("ultralytics").__version__,
    })

# ============================================================
# Aba Processamento
# ============================================================
with tab_app:
    with st.sidebar:
        st.header("⚙️ Parâmetros")
        model_name = st.selectbox("Modelo YOLO (pessoas)", ["yolov8n.pt", "yolov8s.pt"], index=1)
        conf = st.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
        iou = st.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
        save_output = st.checkbox("Salvar MP4", value=True)
        output_path = st.text_input("Caminho do MP4", value="video_output.mp4")
        tracker_cfg = st.text_input("Arquivo do tracker", value="bytetrack.yaml")

    # ------------------ Upload e botão -----------------------
    st.subheader("📤 Envie o vídeo para análise")
    uploaded = st.file_uploader("Selecione o vídeo (mp4, avi, mov, mkv…)", type=None)
    run_button = st.button("📤 Enviar e Analisar Vídeo")

    # ========================================================
    # Correção do PyTorch 2.6+
    # ========================================================
    @st.cache_resource(show_spinner=True)
    def load_model_safely(name: str):
        import torch
        from ultralytics.nn.tasks import DetectionModel
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([DetectionModel])
        if os.path.isfile(name):
            try:
                return YOLO(name)
            except Exception:
                torch.serialization.add_safe_globals([DetectionModel])
                return YOLO(name)
        try:
            return YOLO(name)
        except Exception:
            _clean_ultralytics_cache_for(name)
            return YOLO(name)

    model = load_model_safely(model_name)

    # Layout
    video_col, metrics_col = st.columns([3, 1])
    frame_placeholder = video_col.empty()

    def _save_to_temp(file):
        suffix = os.path.splitext(file.name)[-1] or ".mp4"
        path = f"temp_{int(time.time())}{suffix}"
        with open(path, "wb") as f: f.write(file.read())
        return path

    def _box_center(b): x1, y1, x2, y2 = b; return ((x1 + x2) / 2, (y1 + y2) / 2)
    def _center_in_box(c, b): x, y = c; x1, y1, x2, y2 = b; return x1 <= x <= x2 and y1 <= y <= y2

    # ========================================================
    # Processamento
    # ========================================================
    def process_video(input_path):
        from datetime import datetime
        writer = None
        unique_ids = set()
        frame_count = 0
        start = time.time()
        logs = []

        stream = model.track(
            source=input_path, stream=True, conf=conf, iou=iou, classes=[0],
            tracker=tracker_cfg, persist=True, verbose=False,
        )

        for result in stream:
            frame = result.plot()
            frame_count += 1
            ids, boxes = [], []
            if result.boxes and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().tolist()
                boxes = result.boxes.xyxy.cpu().numpy().tolist()
            persons_ids = [int(i) for i in ids]
            unique_ids.update(persons_ids)

            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, 24.0, (w, h))
            if writer:
                writer.write(frame)
            frame_placeholder.image(frame[:, :, ::-1], channels="RGB")

            fps = frame_count / max(time.time() - start, 1e-6)
            metrics_col.metric("FPS", f"{fps:.1f}")
            metrics_col.metric("Pessoas", str(len(persons_ids)))
            metrics_col.metric("Únicas", str(len(unique_ids)))
            logs.append({"frame": frame_count, "fps": fps, "persons": len(persons_ids)})

        if writer:
            writer.release()
        return {"frames": frame_count, "unique_ids": len(unique_ids), "logs": logs, "out": output_path}

    # ------------------ Execução -----------------------------
    if run_button:
        if uploaded is None:
            st.error("Envie um vídeo primeiro.")
        else:
            in_path = _save_to_temp(uploaded)
            with st.spinner("Processando vídeo..."):
                summary = process_video(in_path)
            st.success(f"✅ {summary['frames']} frames, {summary['unique_ids']} pessoas únicas detectadas.")
            if save_output and summary.get("out"):
                st.video(summary["out"])

# ============================================================
# Aba Dashboard
# ============================================================
with tab_dash:
    import pandas as pd
    st.subheader("📊 Dashboard em tempo real")
    logs = st.session_state.get("rt_logs", [])
    if not logs:
        st.info("Nenhum dado ainda. Rode o processamento na aba 🎬 Processamento.")
    else:
        df = pd.DataFrame(logs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames", int(df["frame"].max()))
        c2.metric("FPS médio", f"{df['fps'].mean():.1f}")
        c3.metric("Máx pessoas", int(df["persons"].max()))
        st.line_chart(df.set_index("frame")[["fps", "persons"]])

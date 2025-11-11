import os, sys, time, glob, pickle, shutil, importlib, subprocess
import numpy as np
import streamlit as st

# ===================== GARANTIR DEPENDÊNCIAS =====================
def ensure(pkg, pip_name=None):
    pip_name = pip_name or pkg
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

def ensure_cv2_headless():
    try:
        import cv2
        if hasattr(cv2, "__file__") and "headless" not in (cv2.__file__ or "").lower():
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless==4.10.0.84"])
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless==4.10.0.84"])

ensure("ultralytics")
ensure("supervision", "supervision==0.21.0")
ensure_cv2_headless()
ensure("numpy", "numpy<2")

import cv2
from ultralytics import YOLO
import supervision as sv
from ultralytics.utils import SETTINGS

# ===================== CONFIG STREAMLIT =====================
st.set_page_config(page_title="Pessoas + EPI Tracker", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #f5f5f5;
        }
        .stApp {
            background-color: #111;
        }
        h1, h2, h3, h4, h5 {
            color: #B22222 !important;
        }
        .stButton>button {
            background-color: #B22222;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #a11e1e;
            color: white;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #222;
            color: #f5f5f5;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #B22222;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== FUNÇÕES AUXILIARES =====================
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", ".ultra_cache")
try:
    os.makedirs(os.environ["ULTRALYTICS_CACHE_DIR"], exist_ok=True)
except Exception:
    pass

def _clean_ultralytics_cache_for(weights_name: str):
    try:
        stem = os.path.splitext(os.path.basename(weights_name))[0]
        dirs = [SETTINGS.get("weights_dir"), os.environ.get("ULTRALYTICS_CACHE_DIR")]
        for d in dirs:
            if d and os.path.isdir(d):
                for p in glob.glob(os.path.join(d, f"{stem}*")):
                    try:
                        if os.path.isfile(p): os.remove(p)
                        elif os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
                    except Exception:
                        pass
    except Exception:
        pass

# ===================== CARREGAMENTO SEGURO DE MODELOS =====================
@st.cache_resource(show_spinner=True)
def load_model_safely(name: str):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([DetectionModel])

    try:
        return YOLO(name)
    except pickle.UnpicklingError:
        _clean_ultralytics_cache_for(name)
        return YOLO(name)
    except Exception as e:
        st.error(f"Falha ao carregar modelo '{name}': {e}")
        st.stop()

# ===================== INTERFACE PRINCIPAL =====================
st.title("🧑‍🏭 Rastreamento de Pessoas + EPIs")
st.caption("Detecção e rastreamento de pessoas, com reconhecimento de EPIs opcionais (capacete, colete, etc.)")

tab_sys, tab_app, tab_dash = st.tabs(["🖥️ Sistema", "🎬 Processamento", "📊 Dashboard"])

# ===================== Aba Sistema =====================
with tab_sys:
    import platform
    st.subheader("Informações do sistema")
    st.json({
        "Python": platform.python_version(),
        "OpenCV": cv2.__version__,
        "Ultralytics": __import__("ultralytics").__version__,
        "NumPy": np.__version__,
    })

# ===================== Aba Processamento =====================
with tab_app:
    st.sidebar.header("⚙️ Parâmetros")
    model_name = st.sidebar.selectbox("Modelo YOLO", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=1)
    conf = st.sidebar.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
    iou = st.sidebar.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
    tracker_cfg = st.sidebar.text_input("Configuração do tracker", "bytetrack.yaml")

    uploaded_video = st.file_uploader("📤 Envie um vídeo para análise", type=["mp4", "avi", "mov", "mkv"])
    analyze_btn = st.button("▶️ Analisar vídeo")

    if analyze_btn:
        if uploaded_video is None:
            st.error("Envie um vídeo primeiro.")
        else:
            temp_path = os.path.join("input_temp.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())

            st.info("🔄 Carregando modelo YOLO...")
            model = load_model_safely(model_name)

            st.info("🎬 Processando vídeo...")
            stream = model.track(source=temp_path, stream=True, conf=conf, iou=iou, classes=[0], tracker=tracker_cfg)

            frame_ph = st.empty()
            frame_count = 0
            for result in stream:
                frame = result.plot()
                frame_count += 1
                frame_ph.image(frame[:, :, ::-1], channels="RGB", caption=f"Frame {frame_count}")

                if frame_count > 300:  # limita para evitar travar Streamlit
                    st.warning("Interrompido após 300 frames (exemplo).")
                    break
            st.success("✅ Processamento concluído!")

# ===================== Aba Dashboard =====================
with tab_dash:
    st.write("📈 Dashboard ainda em construção — mostrará FPS, contagem de pessoas e EPIs detectados.")

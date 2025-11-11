# streamlit_app.py
import os, sys, time, glob, shutil, importlib, subprocess, numpy as np, streamlit as st

# ===================== DEPENDÊNCIAS =====================
def ensure(pkg, pip_name=None):
    pip_name = pip_name or pkg
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

def ensure_cv2_headless():
    try:
        import cv2
        if "headless" not in (cv2.__file__ or "").lower():
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
from ultralytics.utils import SETTINGS
import supervision as sv

# ===================== CONFIG PÁGINA + CSS =====================
st.set_page_config(page_title="Pessoas + EPI Tracker", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #111; color: #f5f5f5; font-family: 'Segoe UI', Roboto, sans-serif; }
h1,h2,h3,h4 { color: #B22222 !important; font-weight: 700; }
section[data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 2px solid #B22222; }
.stButton>button { background-color: #B22222; color: white; border-radius: 8px; font-weight: bold; padding: 0.6em 1.2em; border: none; transition: 0.3s; }
.stButton>button:hover { background-color: #a11e1e; transform: scale(1.05); }
[data-testid="stFileUploaderDropzone"] { background-color: #1a1a1a !important; border: 2px dashed #B22222 !important; color: #ccc !important; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { background-color: #222; color: #f5f5f5; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #B22222 !important; border-bottom: 3px solid #B22222; }
</style>
""", unsafe_allow_html=True)

# ===================== FUNÇÕES AUXILIARES =====================
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", ".ultra_cache")
os.makedirs(os.environ["ULTRALYTICS_CACHE_DIR"], exist_ok=True)

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
                    except: pass
    except: pass

# ===================== CORREÇÃO PYTORCH 2.6 =====================
def load_model_safely(model_name: str):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([DetectionModel])
    try:
        return YOLO(model_name)
    except Exception as e:
        st.warning(f"⚠️ Erro inicial ao carregar {model_name}: {e}")
        _clean_ultralytics_cache_for(model_name)
        try:
            torch.serialization.add_safe_globals([DetectionModel])
            return YOLO(model_name)
        except Exception as e2:
            st.error(f"🚫 Falha crítica ao carregar '{model_name}': {e2}")
            st.stop()

# ===================== INTERFACE =====================
st.title("🧑‍🏭 Rastreamento de Pessoas + EPIs")
st.caption("Detecção e rastreamento de pessoas, com reconhecimento de EPIs opcionais (capacete, colete etc.)")

tab_sys, tab_app, tab_dash = st.tabs(["🖥️ Sistema", "🎬 Processamento", "📊 Dashboard"])

# ---------- Aba Sistema ----------
with tab_sys:
    import platform
    st.subheader("Informações do sistema")
    st.json({
        "Python": platform.python_version(),
        "OpenCV": cv2.__version__,
        "Ultralytics": __import__("ultralytics").__version__,
        "NumPy": np.__version__,
    })

# ---------- Aba Processamento ----------
with tab_app:
    st.sidebar.header("⚙️ Parâmetros")
    model_name = st.sidebar.selectbox("Modelo YOLO", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=1)
    conf = st.sidebar.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
    iou = st.sidebar.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
    tracker_cfg = st.sidebar.text_input("Configuração do tracker", "bytetrack.yaml")

    st.subheader("📤 Envie o vídeo para análise")
    uploaded_video = st.file_uploader("Selecione o vídeo (mp4, avi, mov, mkv…)", type=["mp4", "avi", "mov", "mkv"])
    analyze_btn = st.button("▶️ Enviar e Analisar Vídeo")

    def _save_to_temp(file):
        suffix = os.path.splitext(file.name)[-1] or ".mp4"
        path = f"temp_{int(time.time())}{suffix}"
        with open(path, "wb") as f: f.write(file.read())
        return path

    if analyze_btn:
        if uploaded_video is None:
            st.error("Envie um vídeo primeiro.")
        else:
            temp_path = _save_to_temp(uploaded_video)
            st.info("🔄 Carregando modelo YOLO...")
            model = load_model_safely(model_name)

            st.info("🎬 Processando vídeo...")
            stream = model.track(source=temp_path, stream=True, conf=conf, iou=iou, classes=[0], tracker=tracker_cfg)

            frame_ph = st.empty()
            frame_count, start = 0, time.time()
            for result in stream:
                frame = result.plot()
                frame_count += 1
                frame_ph.image(frame[:, :, ::-1], channels="RGB", caption=f"Frame {frame_count}")
                if frame_count > 300:
                    st.warning("Interrompido após 300 frames (modo demo).")
                    break
            st.success(f"✅ Processamento concluído ({frame_count} frames).")

# ---------- Aba Dashboard ----------
with tab_dash:
    st.info("📊 Dashboard em construção — incluirá FPS médio, contagem de pessoas e EPIs detectados.")

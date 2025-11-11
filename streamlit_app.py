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

try:
    ensure("streamlit_webrtc")
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

import cv2  # noqa
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import supervision as sv

# ============================================================
# Cache dedicado do Ultralytics
# ============================================================
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", ".ultra_cache")
try:
    os.makedirs(os.environ["ULTRALYTICS_CACHE_DIR"], exist_ok=True)
except Exception:
    pass

def _clean_ultralytics_cache_for(weights_name: str):
    try:
        candidates = []
        stem = os.path.splitext(os.path.basename(weights_name))[0]
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
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Pessoas + PPE Track — Streamlit", layout="wide")

# === CSS customizado (tema vermelho e preto) ===
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0b0b; color: #f2f2f2; font-family: 'Segoe UI', Roboto, sans-serif; }
    h1, h2, h3, h4 { color: #e50914 !important; font-weight: 700 !important; letter-spacing: 0.5px; }
    section[data-testid="stSidebar"] { background-color: #141414 !important; border-right: 2px solid #e50914; }
    section[data-testid="stSidebar"] .stButton>button { background-color: #e50914 !important; color: white !important; border: none; border-radius: 4px; font-weight: 600; transition: 0.2s; }
    section[data-testid="stSidebar"] .stButton>button:hover { background-color: #ff1f1f !important; transform: scale(1.05); }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #e50914 !important; }
    .stTabs [aria-selected="true"] { color: #e50914 !important; border-bottom: 3px solid #e50914 !important; font-weight: bold; }
    .stButton>button { background-color: #e50914; color: white; border: none; border-radius: 5px; font-weight: 600; padding: 0.6em 1.2em; transition: 0.3s; }
    .stButton>button:hover { background-color: #ff1f1f; transform: scale(1.03); }
    [data-testid="stFileUploaderDropzone"] { background-color: #1a1a1a !important; border: 2px dashed #e50914 !important; color: #ccc !important; border-radius: 8px; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { background-color: #1a1a1a !important; color: #fff !important; border: 1px solid #e50914 !important; border-radius: 4px !important; }
    .stSlider>div>div>div>div { background: linear-gradient(to right, #e50914, #660000); }
    [data-testid="stMetricValue"] { color: #e50914 !important; font-weight: 700 !important; }
    .stAlert { border-left: 5px solid #e50914 !important; background-color: #1a1a1a !important; color: #fff !important; }
    .stDataFrame { background-color: #0b0b0b !important; border: 1px solid #e50914 !important; border-radius: 6px; }
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-thumb { background: #e50914; border-radius: 4px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "rt_logs" not in st.session_state:
    st.session_state["rt_logs"] = []

st.title("🧑‍🏭 PeopleTrack + PPE — YOLOv8 + ByteTrack (Streamlit)")
st.caption("Detecção + Rastreamento de pessoas; opcionalmente identifica EPIs (modelo PPE separado).")

tab_sys, tab_app, tab_dash = st.tabs(["🖥️ Sistema", "🎬 Processamento", "📊 Dashboard"])

# ======================= Aba Sistema ========================
with tab_sys:
    st.subheader("Informações do Sistema/GPU")
    import platform
    st.write({
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "ultralytics": __import__("ultralytics").__version__,
    })

# ==================== Aba Processamento =====================
with tab_app:
    with st.sidebar:
        st.header("Parâmetros")
        model_name = st.selectbox("Modelo YOLO (pessoas)", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
        conf = st.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
        iou = st.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
        max_frames = st.number_input("Limite de frames (0 = todos)", min_value=0, value=0, step=1)
        save_output = st.checkbox("Salvar resultado em MP4", value=True)
        output_path = st.text_input("Caminho do MP4 de saída", value="video_output.mp4")
        tracker_cfg = st.text_input("Arquivo de tracker (.yaml)", value="bytetrack.yaml")

        st.markdown("---")
        st.subheader("Modelo PPE (opcional)")
        ppe_model_path = st.text_input("Caminho para modelo PPE (ex: ppe_best.pt)", value="")
        ppe_model_classes_input = st.text_input("Classes PPE (vírgulas)", value="helmet,vest")

    run_button = st.button("▶️ Processar vídeo enviado")

    # === CORREÇÃO DO CARREGAMENTO DE MODELOS ===
    @st.cache_resource(show_spinner=True)
    def load_model_safely(name: str):
        if os.path.isfile(name):
            try:
                return YOLO(name)
            except Exception as e:
                st.error(f"Falha ao carregar modelo local '{name}': {e}")
                st.stop()

        official_models = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
        if name not in official_models:
            st.error(f"Modelo '{name}' não encontrado. Use um dos modelos oficiais: {', '.join(official_models)}")
            st.stop()

        try:
            return YOLO(name)
        except Exception as e:
            _clean_ultralytics_cache_for(name)
            try:
                return YOLO(name)
            except Exception as e2:
                st.error(f"Erro ao baixar modelo '{name}': {e2}")
                st.stop()

    model = load_model_safely(model_name)

    ppe_model, ppe_classes = None, []
    if ppe_model_path.strip():
        try:
            ppe_model = load_model_safely(ppe_model_path.strip())
            ppe_classes = [c.strip() for c in ppe_model_classes_input.split(",") if c.strip()]
        except Exception as e:
            st.warning(f"Falha ao carregar modelo PPE: {e}")

    uploaded = st.file_uploader("Envie um vídeo (mp4, avi, mov, mkv…)", type=None)

    video_col, metrics_col = st.columns([3, 1])
    frame_placeholder = video_col.empty()
    track_table_placeholder = metrics_col.empty()
    csv_preview_placeholder = st.empty()

    def _save_to_temp(uploaded_file) -> str:
        suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
        base = os.path.splitext(os.path.basename(uploaded_file.name))[0] or "input"
        path = os.path.join(f"data_cache_input_{base}{suffix}")
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        return path

    def _box_center(box): x1, y1, x2, y2 = box; return ((x1+x2)/2, (y1+y2)/2)
    def _center_in_box(c, b): x,y=c; x1,y1,x2,y2=b; return x1<=x<=x2 and y1<=y<=y2

    def process_video(input_path: str):
        from datetime import datetime
        writer=None; unique_ids=set(); frame_count=0; start_time=time.time(); logs=[]
        stream=model.track(source=input_path,stream=True,conf=conf,iou=iou,classes=[0],tracker=tracker_cfg,persist=True,verbose=False)
        for result in stream:
            frame=result.plot(); frame_count+=1
            ids=[]; boxes=[]
            if result.boxes and result.boxes.id is not None:
                try: ids=result.boxes.id.cpu().numpy().tolist()
                except: ids=[]
                try: boxes=result.boxes.xyxy.cpu().numpy().tolist()
                except: boxes=[]
            persons_ids=[int(i) for i in ids]
            persons_boxes=[[float(b) for b in bx] for bx in boxes]
            unique_ids.update(persons_ids)

            persons_with_ppe={}; ppe_counts=0
            if ppe_model:
                try:
                    ppe_res=ppe_model.predict(frame,conf=conf,iou=iou,verbose=False)[0]
                    xyxy=ppe_res.boxes.xyxy.cpu().numpy().tolist() if getattr(ppe_res,"boxes",None) else []
                    cls_ids=ppe_res.boxes.cls.cpu().numpy().astype(int).tolist() if getattr(ppe_res,"boxes",None) else []
                    model_names=getattr(ppe_model,"names",{})
                    for i,bx in enumerate(xyxy):
                        cid=cls_ids[i] if i<len(cls_ids) else None
                        label=model_names.get(cid,str(cid))
                        ppe_counts+=1
                        c=_box_center(bx)
                        for tid,pb in zip(persons_ids,persons_boxes):
                            if _center_in_box(c,pb):
                                persons_with_ppe.setdefault(tid,{}).setdefault(label,[]).append(bx)
                                break
                except: pass

            logs.append({"frame":frame_count,"persons_no_frame":len(persons_ids),"persons_unicas":len(unique_ids),"ppe_detections":ppe_counts})

            if save_output and writer is None:
                h,w=frame.shape[:2]; fourcc=cv2.VideoWriter_fourcc(*"mp4v"); writer=cv2.VideoWriter(output_path,fourcc,24.0,(w,h))
            if writer: writer.write(frame)
            frame_placeholder.image(frame[:,:,::-1],channels="RGB")

            fps=frame_count/max(time.time()-start_time,1e-6)
            metrics_col.metric("FPS",f"{fps:.1f}")
            metrics_col.metric("Pessoas",str(len(persons_ids)))
            metrics_col.metric("Únicas",str(len(unique_ids)))

            st.session_state["rt_logs"].append({"ts":datetime.utcnow().isoformat(),"frame":frame_count,"fps":float(fps),"persons_no_frame":int(len(persons_ids)),"persons_unicas":int(len(unique_ids))})
            if max_frames and frame_count>=max_frames: break
        if writer: writer.release()
        return {"frames":frame_count,"unique_ids":len(unique_ids),"out":output_path if save_output else None,"logs":logs}

    summary=None
    if run_button:
        if uploaded is None:
            st.error("Envie um vídeo primeiro.")
        else:
            in_path=_save_to_temp(uploaded)
            with st.spinner("Processando vídeo..."):
                summary=process_video(in_path)
            st.success(f"✅ Concluído: {summary['frames']} frames, {summary['unique_ids']} pessoas únicas.")
            if save_output and summary.get("out"): st.video(summary["out"])

# ======================= Aba Dashboard ======================
with tab_dash:
    st.subheader("📊 Dashboard em tempo (quase) real")
    import pandas as pd
    logs=st.session_state.get("rt_logs",[])
    if not logs:
        st.info("Nenhum dado ainda. Rode o processamento na aba 🎬 Processamento.")
    else:
        df=pd.DataFrame(logs)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Frames",int(df["frame"].max()))
        c2.metric("FPS médio",f"{df['fps'].mean():.1f}")
        c3.metric("Pico de pessoas",int(df["persons_no_frame"].max()))
        c4.metric("Pessoas únicas",int(df["persons_unicas"].max()))
        st.line_chart(df.set_index("frame")[["fps","persons_no_frame"]])

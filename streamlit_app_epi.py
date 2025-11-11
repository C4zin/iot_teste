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
# Dependency helpers (keeps behavior similar to original project)
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

# ensure core deps (assumes requirements.txt used in deployment)
ensure("ultralytics")
ensure("supervision", "supervision==0.21.0")
ensure_cv2_headless()
ensure("lapx", "lapx>=0.5.9")
ensure("numpy", "numpy<2")

import cv2  # noqa
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import supervision as sv

# keep ultralytics cache local to avoid collisions
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

# ================= Streamlit UI =================
st.set_page_config(page_title="EPITrack Vision - Streamlit", layout="wide")

if "rt_logs" not in st.session_state:
    st.session_state["rt_logs"] = []

st.title("🦺 EPITrack Vision — Detecção de Pessoas + EPIs (YOLOv8 + ByteTrack)")
st.caption("Use um modelo YOLOv8 treinado para detectar 'person' e classes de EPI (capacete, colete, etc.).")

tab_sys, tab_app, tab_dash = st.tabs(["🖥️ Sistema", "🎬 Processamento", "📊 Dashboard"])

with tab_sys:
    st.subheader("Informações do Sistema/GPU")
    import platform
    st.write({
        "python": platform.python_version(),
        "numpy": np.__version__ if 'np' in globals() else None,
        "opencv": cv2.__version__ if 'cv2' in globals() else None,
        "ultralytics": __import__("ultralytics").__version__,
    })
    import shutil as _shutil, subprocess as _subprocess
    if _shutil.which("nvidia-smi"):
        try:
            out = _subprocess.check_output(["nvidia-smi"], text=True)
            st.code(out)
        except Exception as e:
            st.warning(f"Falha ao executar nvidia-smi: {e}")
    else:
        st.info("GPU NVIDIA não detectada (ou `nvidia-smi` indisponível).")

# ------------------ Processing Tab ------------------
with tab_app:
    with st.sidebar:
        st.header("Parâmetros")
        model_source = st.selectbox("Fonte do modelo", ["Pretrained (yolov8n.pt)", "Custom (upload ou nome)"], index=1)
        custom_weights = None
        if model_source.startswith("Custom"):
            uploaded_weights = st.file_uploader("Envie arquivo de pesos (.pt) ou digite o nome do weights (ex: yolov8n_epi.pt)", type=["pt"], accept_multiple_files=False)
            if uploaded_weights is not None:
                # save uploaded weights to temp path
                wpath = os.path.join("weights_cache", uploaded_weights.name)
                os.makedirs("weights_cache", exist_ok=True)
                with open(wpath, "wb") as f:
                    f.write(uploaded_weights.read())
                custom_weights = wpath
            else:
                custom_weights = st.text_input("ou insira o nome do modelo (ex: yolov8n_epi.pt) / caminho público", value="yolov8n_epi.pt")

        conf = st.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
        iou = st.slider("IoU NMS", 0.1, 0.9, 0.5, 0.05)
        max_frames = st.number_input("Limite de frames (0 = todos)", min_value=0, value=0, step=1)
        save_output = st.checkbox("Salvar resultado em MP4", value=True)
        output_path = st.text_input("Caminho do MP4 de saída", value="video_output_epi.mp4")
        tracker_cfg = st.text_input("Arquivo de tracker (.yaml)", value="bytetrack.yaml")
        overlap_threshold = st.slider("Overlap mínimo p/ considerar EPI presente (%)", 1, 90, 10, 1)
        st.markdown("---")
        if st.button("♻️ Limpar estado"):
            for k in list(st.session_state.keys()):
                try:
                    del st.session_state[k]
                except Exception:
                    pass
            try:
                st.cache_resource.clear()
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    run_button = st.button("▶️ Processar vídeo")

    @st.cache_resource(show_spinner=True)
    def load_model_safely(name: str):
        try:
            return YOLO(name)
        except (pickle.UnpicklingError, RuntimeError, ValueError):
            _clean_ultralytics_cache_for(name)
            return YOLO(name)

    # choose model name
    if model_source.startswith("Pretrained"):
        model_name = "yolov8n.pt"
    else:
        model_name = custom_weights or "yolov8n_epi.pt"

    try:
        model = load_model_safely(model_name)
    except Exception as e:
        st.error(f"Falha ao carregar modelo '{model_name}': {e}")
        st.stop()

    # inspect class names in model (expecting 'person' + EPI classes)
    model_names = getattr(model, "names", None)
    if model_names is None:
        try:
            model_names = model.model.names
        except Exception:
            model_names = None

    if not model_names:
        st.warning("Não foi possível determinar as classes do modelo. Pressione para continuar com detecção geral (pessoas podem não estar presentes).")
        model_names = {0: "person"}

    st.sidebar.write("Classes detectáveis (do modelo):")
    try:
        # display names in order
        names_list = [model_names[i] for i in sorted(model_names.keys())]
        st.sidebar.write(names_list)
    except Exception:
        st.sidebar.write(model_names)

    # Video upload
    uploaded = st.file_uploader(
        "Envie um arquivo de vídeo (mp4, avi, mov, mkv…)", type=None, accept_multiple_files=False
    )

    video_col, metrics_col = st.columns([3, 1])
    frame_placeholder = video_col.empty()
    metrics_col_placeholder = metrics_col.empty()
    csv_preview_placeholder = st.empty()

    def _save_to_temp(uploaded_file) -> str:
        suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
        base = os.path.splitext(os.path.basename(uploaded_file.name))[0] or "input"
        temp_path = os.path.join(f"data_cache_input_{base}{suffix}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        return temp_path

    # utility: compute intersection over area ratio of bbox inside person bbox
    def bbox_iou(boxA, boxB):
        # boxes are [x1,y1,x2,y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxBArea = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
        # return fraction of B inside A (i.e., how much of epi box is inside person box)
        return interArea / boxBArea

    def process_video(input_path: str):
        from datetime import datetime
        writer = None
        frame_count = 0
        start_time = time.time()
        logs = []
        unique_person_ids = set()
        # track all classes (None) so we get persons and EPIs
        stream = model.track(
            source=input_path,
            stream=True,
            conf=conf,
            iou=iou,
            classes=None,
            tracker=tracker_cfg,
            persist=True,
            verbose=False,
        )

        # prepare list of epi class indices (all except 'person' if present)
        epi_class_indices = [int(k) for k,v in model_names.items() if v.lower() != "person"]
        person_class_indices = [int(k) for k,v in model_names.items() if v.lower() == "person"]
        # if no explicit 'person' class, assume COCO person=0 will be used
        if not person_class_indices:
            person_class_indices = [0]

        for result in stream:
            frame = result.plot()
            frame_count += 1

            # parse detections
            detections = []
            if result.boxes is not None:
                try:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy()
                    ids = None
                    try:
                        ids = result.boxes.id.cpu().numpy()
                    except Exception:
                        ids = [None]*len(xyxy)
                    confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else [0.0]*len(xyxy)
                    for (b, c, tid, conf) in zip(xyxy, cls, ids, confs):
                        ci = int(c)
                        name = model_names.get(ci, str(ci))
                        detections.append({
                            "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "cls": ci,
                            "name": name,
                            "id": int(tid) if (tid is not None and int(tid) == tid) else None,
                            "conf": float(conf),
                        })
                except Exception:
                    # best-effort fallback
                    pass

            # separate persons and epis
            persons = [d for d in detections if d["cls"] in person_class_indices]
            epis = [d for d in detections if d["cls"] in epi_class_indices]

            people_with_epi = []
            people_without_epi = []

            for p in persons:
                pid = p.get("id", None)
                if pid is not None:
                    unique_person_ids.add(int(pid))
                has_any_epi = False
                matched_epis = []
                for e in epis:
                    overlap = bbox_iou(p["xyxy"], e["xyxy"])
                    if overlap >= (overlap_threshold/100.0):
                        has_any_epi = True
                        matched_epis.append(e)
                if has_any_epi:
                    people_with_epi.append({"person": p, "matched_epis": matched_epis})
                else:
                    people_without_epi.append({"person": p})

            logs.append({
                "frame": frame_count,
                "people_no_frame": len(persons),
                "people_with_epi": len(people_with_epi),
                "people_without_epi": len(people_without_epi),
                "people_unicas": len(unique_person_ids),
                "detections": [{"cls": d["cls"], "name": d["name"], "id": d.get("id"), "xyxy": d["xyxy"]} for d in detections],
            })

            # write output video
            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, 24.0, (w, h))
            if save_output and writer is not None:
                writer.write(frame)

            # UI updates
            frame_placeholder.image(frame[:, :, ::-1], channels="RGB")
            elapsed = max(time.time() - start_time, 1e-6)
            fps = frame_count / elapsed
            metrics_col.metric("FPS (estimado)", f"{fps:0.1f}")
            metrics_col.metric("Pessoas no frame", str(len(persons)))
            metrics_col.metric("Pessoas com EPI", str(len(people_with_epi)))
            metrics_col.metric("Pessoas sem EPI", str(len(people_without_epi)))
            if len(unique_person_ids) > 0:
                try:
                    sample_ids = sorted(unique_person_ids)[-10:]
                except Exception:
                    sample_ids = list(unique_person_ids)[:10]
                metrics_col_placeholder.write({"IDs rastreadas (amostra)": sample_ids})

            # session logs for dashboard
            st.session_state["rt_logs"].append({
                "ts": datetime.utcnow().isoformat(),
                "frame": frame_count,
                "fps": float(fps),
                "people_no_frame": int(len(persons)),
                "people_with_epi": int(len(people_with_epi)),
                "people_without_epi": int(len(people_without_epi)),
                "people_unicas": int(len(unique_person_ids)),
            })
            if len(st.session_state["rt_logs"]) > 5000:
                st.session_state["rt_logs"] = st.session_state["rt_logs"][-2000:]

            if max_frames and frame_count >= max_frames:
                break

        if writer is not None:
            writer.release()

        return {
            "frames": frame_count,
            "unique_persons": len(unique_person_ids),
            "out": output_path if save_output else None,
            "logs": logs,
        }

    summary = None

    if run_button:
        if uploaded is None:
            st.error("Envie um vídeo primeiro.")
        else:
            in_path = _save_to_temp(uploaded)
            with st.spinner("Processando vídeo (detecção de pessoas + EPIs)..."):
                summary = process_video(in_path)

            st.success(f"Concluído: {summary['frames']} frames • {summary['unique_persons']} pessoas únicas.")

            if save_output and summary.get("out"):
                st.video(summary["out"])

            if summary.get("logs"):
                import pandas as pd
                # flatten logs for CSV: one line per frame with counts
                rows = []
                for r in summary["logs"]:
                    rows.append({
                        "frame": r["frame"],
                        "people_no_frame": r["people_no_frame"],
                        "people_with_epi": r["people_with_epi"],
                        "people_without_epi": r["people_without_epi"],
                        "people_unicas": r["people_unicas"],
                    })
                df_logs = pd.DataFrame(rows)
                csv_preview_placeholder.dataframe(df_logs.head(50), use_container_width=True)
                csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Baixar CSV com métricas",
                    data=csv_bytes,
                    file_name="rastreamento_pessoas_epi.csv",
                    mime="text/csv",
                )

# ------------------ Dashboard Tab ------------------
with tab_dash:
    st.subheader("📊 Dashboard em tempo (quase) real - EPITrack")
    import pandas as pd
    logs = st.session_state.get("rt_logs", [])
    if not logs:
        st.info("Nenhum dado ainda. Rode o processamento na aba **🎬 Processamento**.")
    else:
        df = pd.DataFrame(logs)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames processados", int(df["frame"].max()))
        c2.metric("FPS médio", f"{df['fps'].mean():.1f}")
        c3.metric("Pico de pessoas/frame", int(df["people_no_frame"].max()))
        c4.metric("Pessoas únicas (total)", int(df["people_unicas"].max()))
        st.markdown("---")
        st.write("### Séries temporais")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.caption("FPS por frame")
            st.line_chart(df.set_index("frame")["fps"])
        with tcol2:
            st.caption("Pessoas no frame")
            st.line_chart(df.set_index("frame")["people_no_frame"])
        st.markdown("---")
        st.write("### Últimos eventos")
        st.dataframe(
            df[["frame", "fps", "people_no_frame", "people_with_epi", "people_without_epi", "people_unicas", "ts"]]
              .sort_values("frame", ascending=False)
              .head(25),
            use_container_width=True,
            hide_index=True
        )
        st.markdown("---")
        st.write("### Alertas")
        alerts = []
        if df["fps"].mean() < 12:
            alerts.append("⚠️ **Baixo desempenho**: FPS médio abaixo de 12.")
        if df["people_no_frame"].max() >= 8:
            alerts.append("🚦 **Alta densidade**: pico ≥ 8 pessoas no mesmo frame.")
        last_row = df.sort_values("frame").iloc[-1]
        if last_row["people_no_frame"] == 0:
            alerts.append("ℹ️ **Sem detecções no último frame**.")
        # EPI-specific alert: if many people without EPI
        total_people = df["people_no_frame"].sum()
        total_without = df["people_without_epi"].sum()
        if total_people > 0 and (total_without / max(1, total_people)) > 0.25:
            alerts.append("⚠️ **Muitas pessoas sem EPI**: mais de 25% das detecções sem EPI (acum).")
        if alerts:
            for a in alerts:
                st.write(a)
        else:
            st.success("✅ Nenhum alerta no momento.")

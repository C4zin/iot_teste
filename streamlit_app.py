# streamlit_app.py
import os
import time
import urllib.request
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import cv2

# NOTE: ultralytics must be installed (ultralytics >= 8.0)
from ultralytics import YOLO

st.set_page_config(page_title="EPITrack Vision — Streamlit (All-in-one)", layout="wide")

# ---------------------------
# Config / constantes
# ---------------------------
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-construction-ppe.pt"
MODEL_FILENAME = "yolov8n_epi.pt"
UPLOAD_DIR = "uploads"
OUTPUT_VIDEO = "Analisado.mp4"
OUTPUT_CSV = "results_epi.csv"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------
def download_default_model(model_path: str):
    if os.path.exists(model_path):
        return model_path
    st.info("🔽 Baixando modelo público de EPIs (Construction-PPE) — isso pode levar alguns minutos...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        st.success(f"✅ Modelo salvo em {model_path}")
        return model_path
    except Exception as e:
        st.error(f"Erro ao baixar modelo automático: {e}")
        raise

def bbox_iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxBArea = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return interArea / boxBArea

def ensure_model_available(model_path: str):
    if not os.path.exists(model_path):
        download_default_model(model_path)
    return model_path

def safe_cast_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

# ---------------------------
# UI
# ---------------------------
st.title("🦺 EPITrack Vision — Detecção de Pessoas + EPIs (all-in-one)")
st.markdown("Envie um vídeo e o app processará localmente (YOLOv8). Será gerado `Analisado.mp4` com anotações e `results_epi.csv` com métricas por frame.")

col1, col2 = st.columns([3,1])

with col2:
    st.markdown("### Configurações")
    model_mode = st.selectbox("Modelo", ["Auto-download (Construction-PPE)", "Upload meu .pt"], index=0)
    uploaded_model = None
    custom_model_path = None
    if model_mode.endswith(".pt"):
        uploaded_model = st.file_uploader("Envie arquivo .pt do YOLOv8", type=["pt"])
        if uploaded_model is not None:
            os.makedirs("weights_cache", exist_ok=True)
            custom_model_path = os.path.join("weights_cache", uploaded_model.name)
            with open(custom_model_path, "wb") as f:
                f.write(uploaded_model.read())
            st.success(f"Modelo salvo em {custom_model_path}")

    conf = st.slider("Confiança mínima", 0.1, 0.9, 0.35, 0.05)
    iou_nms = st.slider("IOU NMS", 0.1, 0.9, 0.45, 0.05)
    overlap_pct = st.slider("Overlap mínimo p/ considerar EPI (%)", 1, 90, 10, 1)
    overlap_frac = overlap_pct / 100.0
    max_frames = st.number_input("Limite de frames (0 = todos)", min_value=0, value=0, step=1)
    save_video_checkbox = st.checkbox("Salvar vídeo anotado (Analisado.mp4)", value=True)

with col1:
    uploaded_video = st.file_uploader("Envie um vídeo (mp4, avi, mkv, mov)", type=["mp4","avi","mkv","mov"])

process_btn = st.button("▶️ Processar vídeo")

# ---------------------------
# Main processing
# ---------------------------
if process_btn:
    if uploaded_video is None:
        st.error("Envie um vídeo primeiro.")
        st.stop()

    # Save uploaded video to uploads dir
    input_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success(f"Vídeo salvo em: {input_path}")

    # Choose model
    if custom_model_path:
        model_path = custom_model_path
    else:
        model_path = MODEL_FILENAME

    # ensure model exists (download if needed)
    try:
        ensure_model_available(model_path)
    except Exception as e:
        st.error("Não foi possível obter o modelo. Verifique a conexão e tente novamente.")
        st.stop()

    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Falha ao carregar o modelo: {e}")
        st.stop()

    # Inspect class names
    model_names = getattr(model, "names", None)
    if model_names is None:
        try:
            model_names = model.model.names
        except Exception:
            model_names = {0: "person"}

    # Determine indices
    person_indices = [int(k) for k,v in model_names.items() if v.lower() == "person"]
    if len(person_indices) == 0:
        # fallback to COCO person index 0
        person_indices = [0]
    epi_indices = [int(k) for k in model_names.keys() if int(k) not in person_indices]

    st.info(f"Classes detectáveis: { [model_names[i] for i in sorted(model_names.keys())] }")
    st.info(f"Person indices: {person_indices} | EPI indices: {epi_indices}")

    # Prepare video writer (we will use the frames from the stream result's plot())
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Não foi possível abrir o vídeo para leitura.")
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    temp_output = OUTPUT_VIDEO
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Process with model.track to obtain IDs when available
    stream = model.track(source=input_path, stream=True, conf=conf, iou=iou_nms, classes=None, tracker=None, persist=True, verbose=False)

    frame_idx = 0
    rows = []
    unique_ids = set()
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    # iterate
    try:
        for res in stream:
            frame_idx += 1
            # get plotted frame with boxes
            try:
                frame_vis = res.plot()  # BGR numpy array
            except Exception:
                # fallback to orig_img if plot not available
                frame_vis = getattr(res, "orig_img", None)
                if frame_vis is None:
                    continue

            # parse boxes
            detections = []
            if getattr(res, "boxes", None) is not None:
                try:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    cls_arr = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else [0.0]*len(xyxy)
                    ids_arr = None
                    try:
                        ids_arr = res.boxes.id.cpu().numpy().astype(int)
                    except Exception:
                        ids_arr = [None]*len(xyxy)
                    for b, c, idv, cf in zip(xyxy, cls_arr, ids_arr, confs):
                        detections.append({
                            "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "cls": int(c),
                            "id": int(idv) if (idv is not None and idv == idv) else None,
                            "conf": float(cf),
                            "name": model_names.get(int(c), str(int(c)))
                        })
                except Exception:
                    pass

            # separate persons and epis
            persons = [d for d in detections if d["cls"] in person_indices]
            epis = [d for d in detections if d["cls"] in epi_indices]

            people_with_epi = []
            people_without_epi = []

            for p in persons:
                pid = p.get("id", None)
                if pid is not None:
                    unique_ids.add(int(pid))
                has_any = False
                for e in epis:
                    ov = bbox_iou(p["xyxy"], e["xyxy"])
                    if ov >= overlap_frac:
                        has_any = True
                        break
                if has_any:
                    people_with_epi.append(p)
                else:
                    people_without_epi.append(p)

            # annotate top bar (counts)
            cv2.rectangle(frame_vis, (0,0), (width, 36), (30,30,30), -1)
            txt = f"Frame {frame_idx} | People: {len(persons)} | With EPI: {len(people_with_epi)} | Without EPI: {len(people_without_epi)} | Unique IDs: {len(unique_ids)}"
            cv2.putText(frame_vis, txt, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # write frame
            writer.write(frame_vis)

            rows.append({
                "frame": frame_idx,
                "people_no_frame": len(persons),
                "people_with_epi": len(people_with_epi),
                "people_without_epi": len(people_without_epi),
                "people_unicas": len(unique_ids)
            })

            # update UI progress
            elapsed = time.time() - start_time
            status_text.text(f"Processando frame {frame_idx} — elapsed {elapsed:.1f}s")
            # note: model.track yields until video end; cannot know total frames easily here, approximate using video length
            if max_frames and frame_idx >= max_frames:
                break
    except Exception as e:
        st.error(f"Erro durante processamento: {e}")
    finally:
        writer.release()

    # save CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        st.success(f"CSV salvo em: {OUTPUT_CSV}")
    else:
        df = pd.DataFrame(columns=["frame","people_no_frame","people_with_epi","people_without_epi","people_unicas"])
        st.warning("Nenhum frame processado com sucesso.")

    # show results
    if save_video_checkbox and os.path.exists(OUTPUT_VIDEO):
        st.video(OUTPUT_VIDEO)
        st.success(f"Vídeo anotado salvo em: {OUTPUT_VIDEO}")
    else:
        st.info("Opção de salvar vídeo desmarcada ou arquivo não encontrado.")

    # dashboard: metrics and charts
    st.markdown("---")
    st.header("📊 Dashboard de métricas")

    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames processados", int(df["frame"].max()))
        c2.metric("Pessoas únicas (máx)", int(df["people_unicas"].max()))
        c3.metric("Pico pessoas/frame", int(df["people_no_frame"].max()))
        c4.metric("Total sem EPI (acum.)", int(df["people_without_epi"].sum()))

        st.write("### Séries temporais (amostra)")
        st.line_chart(df.set_index("frame")[["people_no_frame","people_with_epi","people_without_epi"]])

        st.write("### Tabela (primeiras 200 linhas)")
        st.dataframe(df.head(200))

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar CSV com métricas", data=csv_bytes, file_name=OUTPUT_CSV, mime="text/csv")
    else:
        st.info("Nenhuma métrica disponível para mostrar.")

    st.balloons()

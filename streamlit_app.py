# streamlit_app.py
import os
import time
import io
import shutil
import zipfile
import urllib.request
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import cv2

# ultralytics (YOLOv8)
from ultralytics import YOLO

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="EPITrack Vision — All-in-one (improved)", layout="wide")
BASE_DIR = Path(".").resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_VIDEO = BASE_DIR / "Analisado.mp4"
OUTPUT_CSV = BASE_DIR / "results_epi.csv"
THUMB_DIR = BASE_DIR / "thumbs_no_epi"
MODEL_LOCAL = BASE_DIR / "yolov8n_epi.pt"

# Candidate URLs (tries in order)
MODEL_URLS = [
    # known release that often exists
    "https://github.com/ultralytics/yolov8/releases/download/v8.1.0/yolov8n-ppe.pt",
    # fallback older assets
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-construction-ppe.pt",
    # or an alternative name (keep as last resort)
    "https://github.com/ultralytics/yolov8/releases/download/v8.2.0/yolov8n-ppe.pt",
]

# UI styling (small)
st.markdown(
    """
    <style>
    .stProgress > div > div > div {background: #e76f51;}
    .big-kpi {font-size:20px; font-weight:600}
    .small-muted {color: #9aa0a6; font-size:12px}
    .badge {padding:6px 10px; border-radius:8px; color:white; font-weight:600}
    </style>
    """,
    unsafe_allow_html=True,
)

# Helpers
def ensure_dirs():
    UPLOAD_DIR.mkdir(exist_ok=True)
    THUMB_DIR.mkdir(exist_ok=True)

def download_model_try(urls, dest: Path, st_container=None):
    """Try multiple urls until one works."""
    if dest.exists():
        return dest
    last_exc = None
    for u in urls:
        try:
            if st_container:
                st_container.info(f"Baixando modelo de: {u}")
            urllib.request.urlretrieve(u, str(dest))
            if st_container:
                st_container.success(f"Modelo salvo em {dest.name}")
            return dest
        except Exception as e:
            last_exc = e
            if st_container:
                st_container.error(f"Falha ao baixar de {u}: {e}")
    raise RuntimeError(f"Não foi possível baixar o modelo. Último erro: {last_exc}")

def bbox_iou(boxA, boxB):
    # box format [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxBArea = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return interArea / boxBArea

def color_for_percentage(pct):
    # pct between 0 and 1 (proportion without EPI)
    if pct < 0.05:
        return "#2ecc71"  # green
    elif pct < 0.25:
        return "#f1c232"  # yellow-ish
    else:
        return "#e74c3c"  # red

# Layout
st.title("🦺 EPITrack Vision — Detecção de Pessoas + EPIs (visual & dashboard)")
st.write("Envie um vídeo e o app analisará localmente. Será gerado **Analisado.mp4** com anotações, **results_epi.csv** com métricas por frame, e thumbnails das pessoas sem EPI para auditoria.")

col_left, col_right = st.columns([3,1])

with col_right:
    st.header("Configurações")
    model_mode = st.selectbox("Modelo", ["Auto-download (recomendado)", "Fazer upload do meu .pt"])
    uploaded_model = None
    custom_model_path = None
    if model_mode.startswith("Fazer upload"):
        uploaded_model = st.file_uploader("Envie pesos YOLOv8 (.pt)", type=["pt"])
        if uploaded_model:
            custom_model_path = BASE_DIR / "weights_cache" / uploaded_model.name
            os.makedirs(custom_model_path.parent, exist_ok=True)
            with open(custom_model_path, "wb") as f:
                f.write(uploaded_model.read())
            st.success(f"Modelo salvo em: {custom_model_path.name}")

    conf = st.slider("Confiança mínima", 0.10, 0.90, 0.35, 0.05)
    iou_nms = st.slider("IoU NMS", 0.10, 0.90, 0.45, 0.05)
    overlap_pct = st.slider("Overlap mínimo p/ considerar EPI (%)", 1, 90, 15, 1)
    overlap_frac = overlap_pct / 100.0
    max_frames = st.number_input("Limite de frames (0 = todos)", min_value=0, value=0, step=1)
    save_video_flag = st.checkbox("Salvar vídeo anotado (Analisado.mp4)", value=True)

with col_left:
    uploaded_video = st.file_uploader("Envie um vídeo (mp4, avi, mkv, mov)", type=["mp4","avi","mkv","mov"])

process = st.button("▶️ Processar e gerar dashboard")

# Main
ensure_dirs()

if process:
    if not uploaded_video:
        st.error("Por favor, envie um vídeo antes de processar.")
        st.stop()

    # save uploaded video
    input_path = UPLOAD_DIR / uploaded_video.name
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success(f"Vídeo salvo em: {input_path}")

    # determine model path
    if custom_model_path:
        model_path = custom_model_path
    else:
        model_path = MODEL_LOCAL

    # download model if missing
    progress_box = st.empty()
    try:
        if not model_path.exists():
            progress_box.info("🔽 Iniciando download do modelo (pode levar alguns minutos)...")
            download_model_try(MODEL_URLS, model_path, st_container=progress_box)
        else:
            progress_box.info(f"Modelo local encontrado: {model_path.name}")
    except Exception as e:
        progress_box.error(f"Não foi possível obter o modelo automaticamente: {e}")
        st.error("Por favor baixe manualmente um modelo .pt e coloque com o nome 'yolov8n_epi.pt' na pasta do app, ou faça upload via 'Fazer upload do meu .pt'.")
        st.stop()

    # load model
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        st.error(f"Falha ao carregar modelo YOLO: {e}")
        st.stop()

    # read class names
    model_names = getattr(model, "names", None)
    if model_names is None:
        try:
            model_names = model.model.names
        except Exception:
            model_names = {0: "person"}

    # identify indices
    person_indices = [int(k) for k,v in model_names.items() if v.lower() == "person"]
    if not person_indices:
        person_indices = [0]  # fallback
    epi_indices = [int(k) for k in model_names.keys() if int(k) not in person_indices]

    # show detected model classes
    st.info(f"Classes detectáveis: { [model_names[i] for i in sorted(model_names.keys())] }")

    # Prepare video writer
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Não foi possível abrir o vídeo.")
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    cap.release()

    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # tracking with model.track to obtain ids when available
    stream = model.track(source=str(input_path), stream=True, conf=conf, iou=iou_nms, classes=None, persist=True, verbose=False)

    frame_idx = 0
    rows = []
    unique_ids = set()
    thumbs_collected = 0

    total_frames_estimate = int(max(1, cap.get(cv2.CAP_PROP_FRAME_COUNT))) if cap.isOpened() else 0
    pbar = st.progress(0)
    status = st.empty()

    # clear thumbs dir
    if THUMB_DIR.exists():
        shutil.rmtree(THUMB_DIR)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    start_t = time.time()
    try:
        for res in stream:
            frame_idx += 1
            # visual frame
            try:
                vis = res.plot()  # returns BGR numpy image
            except Exception:
                vis = getattr(res, "orig_img", None)
                if vis is None:
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
                    for b,c,idv,cf in zip(xyxy, cls_arr, ids_arr, confs):
                        detections.append({
                            "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "cls": int(c),
                            "id": int(idv) if (idv is not None and idv == idv) else None,
                            "conf": float(cf),
                            "name": model_names.get(int(c), str(int(c)))
                        })
                except Exception:
                    pass

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

            # annotate top bar
            cv2.rectangle(vis, (0,0), (width, 38), (24,24,24), -1)
            txt = f"Frame {frame_idx} | People: {len(persons)} | With EPI: {len(people_with_epi)} | Without EPI: {len(people_without_epi)} | Unique IDs: {len(unique_ids)}"
            cv2.putText(vis, txt, (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

            # write frame
            writer.write(vis)

            # save thumbnails of people without EPI (one per person per frame)
            for idx_p, p in enumerate(people_without_epi):
                x1,y1,x2,y2 = map(int, p["xyxy"])
                # clamp
                x1 = max(0,x1); y1 = max(0,y1); x2 = min(width-1,x2); y2 = min(height-1,y2)
                if x2<=x1 or y2<=y1:
                    continue
                crop = vis[y1:y2, x1:x2]
                tname = THUMB_DIR / f"f{frame_idx}_p{idx_p}.jpg"
                cv2.imwrite(str(tname), crop)
                thumbs_collected += 1

            rows.append({
                "frame": frame_idx,
                "people_no_frame": len(persons),
                "people_with_epi": len(people_with_epi),
                "people_without_epi": len(people_without_epi),
                "people_unicas": len(unique_ids)
            })

            # update UI
            elapsed = time.time() - start_t
            status.text(f"Processando frame {frame_idx} — elapsed {elapsed:.1f}s")
            if total_frames_estimate > 0:
                p = min(1.0, frame_idx/total_frames_estimate)
                pbar.progress(p)
            else:
                # step progress in small increments when no count known
                pbar.progress(min(0.95, frame_idx % 100 / 100.0))

            if max_frames and frame_idx >= max_frames:
                break

    except Exception as e:
        st.error(f"Erro durante processamento: {e}")
    finally:
        writer.release()
        pbar.empty()
        status.empty()

    # postprocess - save CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        st.success(f"CSV salvo em: {OUTPUT_CSV.name}")
    else:
        df = pd.DataFrame(columns=["frame","people_no_frame","people_with_epi","people_without_epi","people_unicas"])
        st.warning("Nenhuma métrica gerada — verifique se o vídeo continha frames e deteções.")

    # show results
    st.markdown("---")
    st.header("📊 Dashboard de resultado")

    # KPIs
    if not df.empty:
        total_frames = int(df["frame"].max())
        total_people_detections = int(df["people_no_frame"].sum())
        total_without = int(df["people_without_epi"].sum())
        total_with = int(df["people_with_epi"].sum())
        unique_count = int(df["people_unicas"].max() if not df["people_unicas"].empty else 0)
        pct_without = total_without / max(1, (total_without + total_with))
        pct_without_display = pct_without * 100.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Frames processados", total_frames)
        k2.metric("Pessoas únicas (estimado)", unique_count)
        k3.metric("Pico pessoas/frame", int(df["people_no_frame"].max()))
        k4.metric("Total sem EPI (acum.)", total_without)

        # percent badge with color
        color = color_for_percentage(pct_without)
        st.markdown(f"<div style='display:flex; gap:12px; align-items:center'><div style='font-weight:700'>Percentual sem EPI (acum.):</div><div class='badge' style='background:{color}'>{pct_without_display:.1f}%</div></div>", unsafe_allow_html=True)

        # time-series chart
        st.write("### Séries temporais")
        st.line_chart(df.set_index("frame")[["people_no_frame","people_with_epi","people_without_epi"]])

        # table
        st.write("### Tabela (amostra)")
        st.dataframe(df.head(200))

        # downloads
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar CSV", data=csv_bytes, file_name=OUTPUT_CSV.name, mime="text/csv")

        if OUTPUT_VIDEO.exists() and save_video_flag:
            st.write("### Vídeo anotado")
            st.video(str(OUTPUT_VIDEO))
            st.success(f"Vídeo anotado salvo: {OUTPUT_VIDEO.name}")

        # prepare thumbs zip
        if THUMB_DIR.exists() and any(THUMB_DIR.iterdir()):
            zip_path = BASE_DIR / "thumbs_no_epi.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for t in sorted(THUMB_DIR.glob("*.jpg")):
                    z.write(t, arcname=t.name)
            with open(zip_path, "rb") as f:
                st.download_button("📥 Baixar thumbnails (pessoas sem EPI)", f.read(), file_name=zip_path.name, mime="application/zip")
        else:
            st.info("Nenhuma pessoa sem EPI detectada (nenhum thumbnail).")

    else:
        st.info("Nenhuma métrica disponível para exibir no dashboard.")

    st.success("Processamento finalizado 🎉")
    st.balloons()

# End of app

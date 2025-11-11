"""
Streamlit app para analisar vídeos e detectar uso de EPIs (PPE).
Funcionalidade:
 - Upload de vídeo
 - Upload opcional de modelo YOLOv8 (.pt) treinado para detectar EPIs (helmet, vest, mask...)
 - Botão "Analisar" para processar vídeo e gerar anotações + relatório
 - Associação de EPIs a pessoas via IoU (sobreposição de caixas)
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from tqdm import tqdm
import math
import ffmpeg
import pandas as pd

# ----- Utilidades -----
def iou(boxA, boxB):
    # boxes format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0
    return interArea / union

def box_xyxy_to_int(box):
    return [int(round(v)) for v in box]

def draw_box(img, box, label, conf, color=(220,50,50), thickness=2):  # color less strong red-ish
    x1,y1,x2,y2 = box_xyxy_to_int(box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    text = f"{label} {conf:.2f}"
    t_size = cv2.getTextSize(text, 0, fontScale=0.5, thickness=1)[0]
    cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# ----- Streamlit UI -----
st.set_page_config(layout="wide", page_title="Análise de EPI (PPE) com IA")
st.title("Analisador de uso de EPI em vídeo — Streamlit + YOLOv8")
st.markdown(
    """
    Envie um vídeo e (opcionalmente) um modelo `YOLOv8 (.pt)` treinado para detectar EPIs (por exemplo: helmet, vest, mask).
    Se não enviar modelo, o app tentará usar um modelo genérico para detectar pessoas apenas.
    """
)

col1, col2 = st.columns([1,2])

with col1:
    st.header("Entradas")
    uploaded_model = st.file_uploader("Modelo YOLO (.pt) — opcional (arraste o arquivo)", type=["pt"])
    uploaded_video = st.file_uploader("Vídeo para analisar", type=["mp4","avi","mov","mkv","webm"])
    st.markdown("**Configurações**")
    conf_thres = st.slider("Limite de confiança", min_value=0.1, max_value=0.95, value=0.35, step=0.05)
    iou_match_thresh = st.slider("IoU mínimo para associar EPI à pessoa", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
    imgsz = st.selectbox("Tamanho de inferência (maior = mais preciso/lento)", options=[320, 480, 640, 800], index=2)
    analyze_button = st.button("🔎 Analisar vídeo")

with col2:
    st.header("Saída")
    output_placeholder = st.empty()
    st.markdown("Resultados e vídeo anotado aparecerão aqui após a análise.")

# ----- Main processing -----
if analyze_button:
    if uploaded_video is None:
        st.warning("Por favor envie um vídeo antes de clicar em 'Analisar'.")
    else:
        # salvar arquivos temporários
        tmp_dir = Path(tempfile.mkdtemp(prefix="ppe_analysis_"))
        video_path = tmp_dir / "input_video"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        model_path = None
        if uploaded_model is not None:
            model_path = tmp_dir / "custom_model.pt"
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.info("Modelo carregado com sucesso. Usando modelo personalizado.")
        else:
            st.info("Nenhum modelo customizado enviado — usando modelo YOLOv8 padrão (yolov8n.pt).")

        progress = st.progress(0)
        status_text = st.empty()

        # load model
        try:
            if model_path is not None:
                model = YOLO(str(model_path))
            else:
                model = YOLO("yolov8n.pt")  # exige internet/pacote já baixado ou cache local
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
            raise

        # identificar nomes de classes do modelo carregado
        model_names = model.names if hasattr(model, "names") else {}
        # classes alvo que indicam EPI (tentativa de mapear automaticamente)
        target_keywords = ["helmet", "hardhat", "hard hat", "vest", "safety vest", "mask", "face_mask", "ppe"]
        # map model classes to normalized names
        class_map = {}
        for idx, name in model_names.items():
            lname = name.lower()
            matched = None
            for kw in target_keywords:
                if kw in lname:
                    # normalize
                    if "helmet" in kw or "hard" in kw:
                        matched = "helmet"
                    elif "vest" in kw:
                        matched = "vest"
                    elif "mask" in kw:
                        matched = "mask"
                    elif "ppe" in kw:
                        matched = "ppe"
            if matched:
                class_map[idx] = matched
        # check if model contains 'person' class
        person_class_ids = [cid for cid, n in model_names.items() if n.lower() in ("person","people","person.0","person0")]
        has_person = len(person_class_ids) > 0

        # abrir vídeo
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None

        # preparar writer para vídeo de saída
        out_vid_path = tmp_dir / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (w,h))

        # estatísticas
        per_frame_reports = []
        cumulative_people = 0
        cumulative_with = {"helmet":0, "vest":0, "mask":0}
        cumulative_counts = {"total_people":0}

        processed = 0
        pbar_total = total_frames if total_frames is not None and total_frames > 0 else 100

        status_text.text("Iniciando processamento de frames...")
        frame_idx = 0
        # iterar frames
        with st.spinner("Processando vídeo — isto pode demorar dependendo do tamanho/modelo..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # inference
                results = model.predict(source=[frame], imgsz=imgsz, conf=conf_thres, verbose=False)  # batch size 1
                # results[0] é o resultado para nosso frame
                res = results[0]
                detections = []
                # cada detecção: xyxy, conf, cls
                if hasattr(res, "boxes") and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()  # Nx4
                    scores = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                    for b, s, c in zip(boxes, scores, classes):
                        detections.append({"box": b.tolist(), "conf": float(s), "cls": int(c), "name": model_names.get(c, str(c))})

                # separar pessoas e EPIs
                people = []
                epis = []
                for d in detections:
                    if has_person and d["cls"] in person_class_ids:
                        people.append(d)
                    else:
                        # se classe mapeada para EPI (via class_map) ou classe tem nome com keyword
                        mapped = class_map.get(d["cls"], None)
                        if mapped:
                            epis.append({**d, "ppe_type": mapped})
                        else:
                            # tentar heurística por nome
                            lname = d["name"].lower()
                            ptype = None
                            if "helmet" in lname or "hard" in lname:
                                ptype = "helmet"
                            elif "vest" in lname:
                                ptype = "vest"
                            elif "mask" in lname or "face" in lname:
                                ptype = "mask"
                            if ptype:
                                epis.append({**d, "ppe_type": ptype})
                            else:
                                # classe desconhecida - ignorar para associação
                                pass

                # se não houver pessoas detectadas, podemos forçar contagem zero ou usar heurística:
                frame_people_count = len(people) if len(people)>0 else 0
                cumulative_counts["total_people"] += frame_people_count

                # para cada pessoa, verificar EPIs por IoU sobrepostos
                person_reports = []
                for p in people:
                    p_box = p["box"]
                    person_has = {"helmet": False, "vest": False, "mask": False}
                    for e in epis:
                        e_box = e["box"]
                        the_iou = iou(p_box, e_box)
                        if the_iou >= iou_match_thresh:
                            person_has[e["ppe_type"]] = True
                    # marcar cumulativos
                    for k in person_has:
                        if person_has[k]:
                            cumulative_with[k] += 1
                    person_reports.append({"person_box": p_box, "has": person_has})

                # anotação do frame
                vis = frame.copy()
                # desenhar people
                for p in people:
                    draw_box(vis, p["box"], label="person", conf=p["conf"], color=(70,130,180), thickness=2)  # steel blue for people
                # desenhar EPIs
                for e in epis:
                    ptype = e["ppe_type"]
                    col = (220,50,50) if ptype in ("helmet","mask") else (40,180,40)  # slightly different colors for types
                    draw_box(vis, e["box"], label=f"{ptype}", conf=e["conf"], color=col, thickness=2)

                # desenhar texto resumo rápido
                y0 = 20
                lines = [
                    f"Frame: {frame_idx}",
                    f"Pessoas detectadas neste frame: {len(people)}",
                    f"EPIs detectados (total neste frame): {len(epis)}"
                ]
                for i, line in enumerate(lines):
                    cv2.putText(vis, line, (10, y0 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                writer.write(vis)

                # update progress
                processed += 1
                if total_frames:
                    progress.progress(min(int((processed/total_frames)*100), 100))
                else:
                    # approximate progress
                    progress.progress(min(int((processed / pbar_total) * 100), 100))
                status_text.text(f"Processando frame {frame_idx} — pessoas: {len(people)} — epis: {len(epis)}")

            # fim loop
        cap.release()
        writer.release()
        progress.progress(100)
        status_text.text("Processamento concluído.")

        # calcular métricas finais
        # observação: contagens cumulativas por frame podem contar a mesma pessoa em frames diferentes; ideal é fazer tracking para estatísticas por pessoa única.
        total_people_frames = cumulative_counts["total_people"]
        metrics = {}
        for k in cumulative_with:
            metrics[k] = {
                "count_frames_with_ppe": cumulative_with[k],
                "percent_over_frames": (cumulative_with[k] / total_people_frames * 100) if total_people_frames>0 else 0.0
            }

        # mostrar resultados
        st.subheader("Resumo (por frames — sem tracking de identidade)")
        df = pd.DataFrame([
            {"PPE": k, "Frames com PPE (somatória)": cumulative_with[k],
             "Percentual (por frames com pessoas)": metrics[k]["percent_over_frames"]}
            for k in cumulative_with
        ])
        st.dataframe(df)

        st.success("Vídeo anotado pronto — veja abaixo e faça download se desejar.")
        # exibir vídeo anotado
        try:
            with open(out_vid_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button("⬇️ Baixar vídeo anotado", data=video_bytes, file_name="annotated_ppe.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Não foi possível exibir/baixar o vídeo anotado: {e}")
            st.write(f"Arquivo salvo em: {out_vid_path}")

        # limpeza opcional: manter arquivos em tmp_dir para inspeção
        st.info(f"Arquivos temporários em: {tmp_dir} (remova manualmente para liberar espaço se desejar).")

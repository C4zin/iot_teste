#!/usr/bin/env python3
"""
detect_epi_video.py
Processa um vídeo para detectar pessoas e EPIs usando um modelo YOLOv8 treinado em Construction-PPE (ou outro custom .pt).
Gera:
 - Analisado.mp4   -> vídeo anotado
 - results_epi.csv -> CSV com uma linha por frame: frame, people_no_frame, people_with_epi, people_without_epi, unique_persons (tracked)
Uso:
    python detect_epi_video.py --input meu_video.mp4 --output Analisado.mp4 --csv results_epi.csv --model yolov8n_epi.pt
Se --model não existir, o script tentará baixar um modelo público Construction-PPE.
"""

import os
import argparse
import urllib.request
import time
from collections import defaultdict

def ensure_deps():
    try:
        import ultralytics, cv2, numpy as np, pandas as pd
    except Exception as e:
        raise RuntimeError("Bibliotecas não instaladas. Rode: pip install ultralytics opencv-python-headless numpy pandas") from e

def download_default_model(model_path):
    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-construction-ppe.pt"
    if not os.path.exists(model_path):
        print("🔽 Baixando modelo público de EPIs (Construction-PPE da Ultralytics)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("✅ Modelo salvo em:", model_path)
        except Exception as e:
            print("❌ Falha ao baixar modelo automático:", e)
            raise

def bbox_iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxBArea = max(1e-6, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return interArea / boxBArea

def main():
    ensure_deps()
    import cv2, numpy as np, pandas as pd
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", "-o", default="Analisado.mp4", help="Caminho do vídeo anotado de saída")
    parser.add_argument("--csv", default="results_epi.csv", help="CSV de saída com métricas por frame")
    parser.add_argument("--model", default="yolov8n_epi.pt", help="Arquivo de pesos YOLOv8 (.pt). Será baixado automaticamente se não existir.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confiança mínima")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS")
    parser.add_argument("--overlap", type=float, default=0.10, help="Fraction of epi box inside person bbox to count as worn (0.1 = 10%)")
    parser.add_argument("--maxframes", type=int, default=0, help="Limite frames (0 = todos)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        download_default_model(args.model)

    model = YOLO(args.model)
    model_names = getattr(model, "names", None) or model.model.names
    # indices
    person_idx = None
    for k,v in model_names.items():
        if v.lower() == "person":
            person_idx = int(k)
            break
    if person_idx is None:
        # assume COCO 0
        person_idx = 0

    epi_indices = [int(k) for k,v in model_names.items() if int(k) != person_idx]

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir vídeo: " + args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame = 0
    unique_ids = set()
    rows = []

    print("Iniciando processamento...")
    start_time = time.time()
    # use model.track to get ids if available
    stream = model.track(source=args.input, stream=True, conf=args.conf, iou=args.iou, classes=None, persist=True, verbose=False)

    for res in stream:
        frame += 1
        img = res.orig_img if hasattr(res, "orig_img") else res.plot()
        # parse boxes
        detections = []
        if res.boxes is not None:
            try:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else [0.0]*len(xyxy)
                ids = None
                try:
                    ids = res.boxes.id.cpu().numpy().astype(int)
                except Exception:
                    ids = [None]*len(xyxy)
                for b,c,idv,cf in zip(xyxy, cls, ids, confs):
                    detections.append({"xyxy":[float(b[0]),float(b[1]),float(b[2]),float(b[3])],"cls":int(c),"id":int(idv) if idv is not None and idv==idv else None,"conf":float(cf)})
            except Exception:
                pass

        persons = [d for d in detections if d["cls"]==person_idx]
        epis = [d for d in detections if d["cls"] in epi_indices]

        people_with = []
        people_without = []
        for p in persons:
            pid = p.get("id", None)
            if pid is not None:
                unique_ids.add(int(pid))
            has = False
            for e in epis:
                ov = bbox_iou(p["xyxy"], e["xyxy"])
                if ov >= args.overlap:
                    has = True
                    break
            if has:
                people_with.append(p)
            else:
                people_without.append(p)

        # annotate image (draw boxes + labels)
        vis = res.plot() if hasattr(res, "plot") else img
        # add a top-bar with counts
        cv2.rectangle(vis, (0,0), (width,40), (20,20,20), -1)
        txt = f"Frame {frame} | People: {len(persons)} | With EPI: {len(people_with)} | Without EPI: {len(people_without)} | Unique IDs: {len(unique_ids)}"
        cv2.putText(vis, txt, (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        writer.write(vis)

        rows.append({
            "frame": frame,
            "people_no_frame": len(persons),
            "people_with_epi": len(people_with),
            "people_without_epi": len(people_without),
            "people_unicas": len(unique_ids)
        })

        if args.maxframes and frame>=args.maxframes:
            break

    writer.release()
    cap.release()
    elapsed = time.time()-start_time
    print(f"Processamento concluído. {frame} frames em {elapsed:.1f}s ({frame/elapsed:.1f} FPS).")
    # save csv
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    print("CSV salvo em:", args.csv)
    print("Vídeo anotado salvo em:", args.output)


if __name__ == "__main__":
    main()

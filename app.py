import streamlit as st
import tempfile
import os
import traceback
import time
import numpy as np
import cv2

st.set_page_config(page_title="Analisador de EPI (PPE)", layout="wide", page_icon="🦺")
st.title("🦺 Analisador de EPI (Streamlit Cloud friendly)")
st.markdown("Envie um vídeo (.mp4, .mov, .avi). O app usa YOLO (Ultralytics) para detectar pessoas/EPIs. Se o modelo não carregar, o app avisará.")

# -----------------------
# Carregamento lazy do modelo (com cache)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model_safe(model_name="yolov8n.pt"):
    """
    Tenta importar ultralytics e carregar o modelo.
    Retorna (model, error_msg). model será None se houver falha.
    """
    try:
        # import dentro da função para evitar import no topo (evita erros de import em alguns ambientes)
        from ultralytics import YOLO
    except Exception as e:
        return None, f"Falha ao importar ultralytics: {e}"

    try:
        model = YOLO(model_name)
        return model, None
    except Exception as e:
        return None, f"Falha ao carregar modelo '{model_name}': {e}"

# -----------------------
# UI: upload e parâmetros
# -----------------------
uploaded_file = st.file_uploader("Envie um vídeo para análise (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
model_name_input = st.text_input("Modelo YOLO (arquivo local ou nome pré-treinado)", value="yolov8n.pt")
sample_rate = st.number_input("Mostrar preview a cada N frames (1 = todo frame)", min_value=1, max_value=60, value=3, step=1)
confidence = st.slider("Confiança mínima (model)", 0.1, 0.95, 0.35, step=0.05)

# status
status = st.empty()
progress_bar = st.progress(0)
preview_box = st.empty()

# -----------------------
# Função principal
# -----------------------
def analyze_video(input_path, output_path, model_name, sample_rate=3, conf_thres=0.35):
    # Carrega modelo de maneira segura
    model, err = load_model_safe(model_name)
    if model is None:
        return {"error": err}

    # Ajusta configurações do modelo se possível
    try:
        # caso a API permita setar conf (ultralytics aceita conf no predict/call)
        pass
    except Exception:
        pass

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "Não foi possível abrir o arquivo de vídeo."}

    # extrair propriedades
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # writer para vídeo anotado
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processed = 0
    show_frame_idx = 0
    detections_counter = {"person_frames": 0, "ppe_frames": 0, "total_frames": 0}
    last_preview_time = 0

    # loop principal
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # checagens de segurança no frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            # pular frame inválido
            continue

        detections_counter["total_frames"] += 1

        # inferência segura por frame
        try:
            # passamos conf na chamada (ultralytics aceita model(frame, conf=...))
            results = model(frame, conf=conf_thres, verbose=False)
        except Exception as e:
            # se erro na inferência de um frame, registra e pula
            # não abortar toda a análise
            print("Erro na inferência de frame:", e)
            traceback.print_exc()
            writer.write(frame)  # escreve frame cru para manter duração
            processed += 1
            # atualizar progresso
            if frame_count > 0:
                progress_bar.progress(min(processed / frame_count, 1.0))
            continue

        # results[0] existe: desenhar/plotar
        try:
            annotated = results[0].plot()  # BGR image
        except Exception:
            # fallback: se plot der errado, usa frame original
            annotated = frame.copy()

        # contadores simples (heurística): conta frames com "person" e frames com keywords de EPI
        try:
            names = results[0].names if hasattr(results[0], "names") else {}
            has_person = False
            has_ppe = False
            if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
                classes = getattr(results[0].boxes, "cls", None)
                if classes is not None:
                    cls_array = classes.cpu().numpy() if hasattr(classes, "cpu") else np.array(classes)
                    for c in cls_array:
                        label = names.get(int(c), str(int(c))).lower()
                        if "person" in label:
                            has_person = True
                        if any(k in label for k in ["helmet", "hardhat", "vest", "mask", "ppe", "glove"]):
                            has_ppe = True
            if has_person:
                detections_counter["person_frames"] += 1
            if has_ppe:
                detections_counter["ppe_frames"] += 1
        except Exception:
            # ignora problemas no parsing de resultados
            pass

        # escreve no vídeo anotado
        try:
            writer.write(annotated)
        except Exception:
            # se escrever falhar, escreve frame original
            writer.write(frame)

        processed += 1

        # atualizar preview (amostragem)
        if (processed % sample_rate) == 0:
            try:
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                # limitar updates ao front-end (não inundar)
                now = time.time()
                if now - last_preview_time > 0.05:  # espaçar updates
                    preview_box.image(frame_rgb, channels="RGB", use_container_width=True)
                    last_preview_time = now
            except Exception:
                pass

        # atualizar barra de progresso
        if frame_count > 0:
            progress = min(processed / frame_count, 1.0)
            progress_bar.progress(progress)

    # finaliza
    cap.release()
    writer.release()
    progress_bar.progress(1.0)

    return {
        "processed_frames": processed,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "counters": detections_counter,
        "error": None
    }

# -----------------------
# Lógica de execução quando o usuário envia um arquivo
# -----------------------
if uploaded_file is not None:
    # salva temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in_path = tmp_in.name

    out_path = os.path.join(tempfile.gettempdir(), f"annotated_{os.path.basename(tmp_in_path)}")

    status.info("🔍 Iniciando análise. Isso pode demorar — aguarde até ver 'Análise concluída'.")

    # executar análise
    result = analyze_video(tmp_in_path, out_path, model_name_input, sample_rate=sample_rate, conf_thres=confidence)

    # tratar resultado
    if result.get("error"):
        st.error(f"Erro: {result['error']}")
        st.write("Detalhes de debug (console): verifique logs do app.")
    else:
        st.success(f"✅ Análise concluída — frames processados: {result['processed_frames']}")
        counters = result.get("counters", {})
        st.metric("Frames totais", result.get("frame_count") or result.get("processed_frames"))
        st.metric("Frames com pessoa (heur.)", counters.get("person_frames", 0))
        st.metric("Frames com EPI (heur.)", counters.get("ppe_frames", 0))

        # mostra vídeo final (se existe)
        try:
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button("⬇️ Baixar vídeo anotado", data=video_bytes, file_name="annotated_video.mp4", mime="video/mp4")
        except Exception as e:
            st.warning(f"Vídeo anotado não pôde ser exibido: {e}")
            st.write(f"Arquivo anotado salvo em: {out_path}")

    # limpar arquivos temporários locais (opcional)
    try:
        os.remove(tmp_in_path)
    except Exception:
        pass

else:
    st.info("⬆️ Envie um vídeo para iniciar a análise.")

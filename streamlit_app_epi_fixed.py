import os
import urllib.request

# ============================================================
# Download automático do modelo público YOLOv8 Construction-PPE
# ============================================================
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-construction-ppe.pt"
MODEL_PATH = "yolov8n_epi.pt"

if not os.path.exists(MODEL_PATH):
    print("🔽 Baixando modelo público de EPIs (Construction-PPE da Ultralytics)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ Modelo salvo em {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erro ao baixar modelo público: {e}")

# ============================================================
# Importa o app principal (com lógica de detecção de EPIs)
# ============================================================
import streamlit_app_epi

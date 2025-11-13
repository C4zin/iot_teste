import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2

st.set_page_config(page_title="Bem-Estar IA", page_icon="🧠", layout="wide")

# ---------------- CARREGAR MODELO ----------------
@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_model.h5")
    return model

emotion_model = load_emotion_model()

emotion_labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

# ---------------- FUNÇÃO DE ANÁLISE ----------------
def detectar_emocao(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (48, 48))
    img_resized = img_resized.astype("float") / 255.0
    img_resized = img_to_array(img_resized)
    img_resized = np.expand_dims(img_resized, axis=0)

    preds = emotion_model.predict(img_resized)[0]
    emotion = emotion_labels[np.argmax(preds)]
    return emotion

# ---------------- INTERFACE ----------------
st.title("🧠 Bem-Estar IA")
st.write("Aplicativo de análise emocional e hábitos usando Deep Learning.")

foto = st.file_uploader("Envie uma foto do seu rosto:", type=["png", "jpg", "jpeg"])

if foto:
    img = Image.open(foto)
    st.image(img, caption="Imagem enviada", use_container_width=True)

    if st.button("Analisar emoção"):
        img_np = np.array(img)
        emotion = detectar_emocao(img_np)
        st.success(f"Emoção detectada: **{emotion}**")

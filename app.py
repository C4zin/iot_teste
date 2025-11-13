import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------------------------------------
# CONFIGURAÇÃO
# -------------------------------------------------------------
st.set_page_config(
    page_title="Bem-Estar IA",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0f0f0f;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CARREGAR MODELO
# -------------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_model.h5")
    return model

emotion_model = load_emotion_model()
emotion_labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]


# -------------------------------------------------------------
# FUNÇÕES
# -------------------------------------------------------------
def detectar_emocao(image):
    # Se vier RGBA → converter
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Converter para BGR (OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Cinza
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Redimensionar 48x48
    img_resized = cv2.resize(img_gray, (48, 48))

    # Normalizar
    img_resized = img_resized.astype("float32") / 255.0

    # Ajustar shape (1, 48, 48, 1)
    img_resized = np.expand_dims(img_resized, axis=-1)
    img_resized = np.expand_dims(img_resized, axis=0)

    preds = emotion_model.predict(img_resized)[0]
    emotion = emotion_labels[np.argmax(preds)]
    return emotion, preds


def sugestoes_emocao(emocao):
    base = {
        "Feliz": [
            "Mantenha os hábitos que estão funcionando!",
            "Aproveite o bom momento para iniciar um novo hábito saudável.",
            "Compartilhe algo positivo com alguém hoje."
        ],
        "Triste": [
            "Tente reservar 30 minutos para algo que você gosta.",
            "Faça pausas a cada 60 minutos para aliviar a mente.",
            "Caso esteja triste por muitos dias, considere conversar com um profissional."
        ],
        "Raiva": [
            "Respire fundo por 5 segundos quando sentir irritação.",
            "Evite ambientes muito estressantes hoje.",
            "Atividade física leve pode ajudar a aliviar a tensão."
        ],
        "Medo": [
            "Liste preocupações e tente resolvê-las uma por uma.",
            "Evite excesso de telas antes de dormir.",
            "Pratique exercícios respiratórios por 2 minutos."
        ],
        "Surpreso": [
            "Organize sua agenda para evitar imprevistos.",
            "Mantenha horários fixos para refeições.",
            "Faça uma pausa rápida e reorganize seu dia."
        ],
        "Nojo": [
            "Tente dividir tarefas desagradáveis ao longo da semana.",
            "Recompense-se após atividades difíceis.",
            "Avalie se não está sobrecarregado."
        ],
        "Neutro": [
            "Inclua 10 minutos de lazer hoje.",
            "Defina uma mini-meta simples para completar.",
            "Beba água e faça alongamentos durante o dia."
        ]
    }
    return base.get(emocao, ["Cuide-se e mantenha uma rotina equilibrada."])


def analisar_rotina(sono, trabalho, lazer, exercicio):
    feedback = []

    if sono < 7:
        feedback.append("Você dormiu pouco. O ideal é entre 7 e 8 horas.")
    elif sono > 9:
        feedback.append("Sono acima da média. Veja se não está ligado a fadiga.")
    else:
        feedback.append("Seu sono está bem regulado!")

    if trabalho > 9:
        feedback.append("Muitas horas de trabalho. Tire pausas estratégicas.")
    else:
        feedback.append("Boa carga de trabalho/estudo.")

    if lazer < 1:
        feedback.append("Pouco lazer detectado. Reserve tempo para você.")
    else:
        feedback.append("Ótimo! Você está se divertindo também.")

    if exercicio == 0:
        feedback.append("Tente incluir ao menos 10 minutos de caminhada.")
    else:
        feedback.append("Manter atividades físicas é excelente!")

    return feedback


# -------------------------------------------------------------
# INTERFACE
# -------------------------------------------------------------
st.title("🧠 Bem-Estar IA")
st.write("Aplicativo de análise emocional e hábitos usando Deep Learning.")

tabs = st.tabs(["📸 Análise de Emoções", "📆 Avaliação de Rotina"])


# =============================================================
# ABA 1 — EMOÇÕES
# =============================================================
with tabs[0]:
    st.subheader("Envie uma foto do seu rosto")

    foto = st.file_uploader("Formatos aceitos: PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])

    if foto:
        img = Image.open(foto)
        st.image(img, use_container_width=True)

        if st.button("Analisar emoções"):
            with st.spinner("Processando..."):
                img_np = np.array(img)

                emocao, probs = detectar_emocao(img_np)

                st.success(f"🎭 Emoção detectada: **{emocao}**")

                st.markdown("### 🔍 Probabilidades:")
                for label, p in zip(emotion_labels, probs):
                    st.write(f"- **{label}** → {p*100:.2f}%")

                st.markdown("### 💡 Recomendações para você:")
                for dica in sugestoes_emocao(emocao):
                    st.markdown(f"- {dica}")


# =============================================================
# ABA 2 — ROTINA
# =============================================================
with tabs[1]:
    st.subheader("Como está sua rotina hoje?")

    sono = st.slider("Horas de sono", 0, 12, 7)
    trabalho = st.slider("Horas de trabalho/estudo", 0, 14, 8)
    lazer = st.slider("Horas de lazer", 0, 8, 1)
    exercicio = st.slider("Horas de exercício", 0, 4, 0)

    if st.button("Analisar rotina"):
        st.markdown("### 📊 Resultados da sua rotina:")

        feedback = analisar_rotina(sono, trabalho, lazer, exercicio)

        for f in feedback:
            st.markdown(f"- {f}")

        st.markdown("### ✨ Dica final:")
        st.write(
            "Registrar diariamente seus hábitos ajuda a monitorar sua evolução "
            "e construir uma rotina mais equilibrada."
        )

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# -------------------------------------------------------------
# CONFIG STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="Bem-Estar IA",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0f0f0f; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# CRIAR MODELO DE EMOÇÕES (SE NÃO EXISTIR)
# -------------------------------------------------------------
def criar_modelo_emocoes():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -------------------------------------------------------------
# CARREGAR OU CRIAR MODELO AUTOMATICAMENTE
# -------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    if not os.path.exists("emotion_tf2.h5"):
        model = criar_modelo_emocoes()
        model.save("emotion_tf2.h5")
    else:
        model = tf.keras.models.load_model("emotion_tf2.h5")
    return model

emotion_model = carregar_modelo()
emotion_labels = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]


# -------------------------------------------------------------
# FUNÇÃO DE DETECÇÃO DE EMOÇÃO
# -------------------------------------------------------------
def detectar_emocao(image):
    # Se vier RGBA → converter
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # RGB → BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Cinza
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 48x48
    img_resized = cv2.resize(img_gray, (48, 48))

    # Normalizar
    img_resized = img_resized.astype("float32") / 255.0

    # Shape (1,48,48,1)
    img_resized = np.expand_dims(img_resized, axis=-1)
    img_resized = np.expand_dims(img_resized, axis=0)

    preds = emotion_model.predict(img_resized)[0]
    emotion = emotion_labels[np.argmax(preds)]

    return emotion, preds


# -------------------------------------------------------------
# SUGESTÕES BASEADAS NA EMOÇÃO
# -------------------------------------------------------------
def sugestoes_emocao(emocao):
    base = {
        "Feliz": [
            "Mantenha os bons hábitos!",
            "Aproveite para iniciar um novo hábito positivo.",
            "Compartilhe algo positivo com alguém."
        ],
        "Triste": [
            "Separe 30 minutos para algo que te faça bem.",
            "Faça pausas durante o dia.",
            "Se permanecer triste, considere falar com alguém de confiança."
        ],
        "Raiva": [
            "Tente exercícios de respiração.",
            "Evite ambientes muito estressantes.",
            "Uma caminhada leve pode ajudar."
        ],
        "Medo": [
            "Liste suas preocupações.",
            "Evite telas antes de dormir.",
            "Faça respiração profunda por 2 minutos."
        ],
        "Surpreso": [
            "Reorganize sua agenda.",
            "Tente manter horários mais fixos.",
            "Faça uma pausa rápida."
        ],
        "Nojo": [
            "Divida tarefas desagradáveis ao longo do dia.",
            "Recompense-se após tarefas difíceis.",
            "Tire pequenas pausas."
        ],
        "Neutro": [
            "Inclua algo divertido no seu dia.",
            "Defina uma mini-meta simples.",
            "Hidrate-se e alongue-se."
        ]
    }
    return base.get(emocao, ["Cuide-se e mantenha equilíbrio."])


# -------------------------------------------------------------
# ANÁLISE DE ROTINA
# -------------------------------------------------------------
def analisar_rotina(sono, trabalho, lazer, exercicio):
    feedback = []

    if sono < 7:
        feedback.append("Você dormiu pouco. O ideal é 7 a 8 horas.")
    elif sono > 9:
        feedback.append("Sono acima da média. Pode ser cansaço acumulado.")
    else:
        feedback.append("Seu sono está equilibrado!")

    if trabalho > 9:
        feedback.append("Carga alta de trabalho. Faça pausas estratégicas.")
    else:
        feedback.append("Boa quantidade de trabalho/estudo.")

    if lazer < 1:
        feedback.append("Pouco lazer. Inclua atividades que te fazem bem.")
    else:
        feedback.append("Ótimo! Você reservou tempo para lazer.")

    if exercicio == 0:
        feedback.append("Tente ao menos 10 minutos de caminhada hoje.")
    else:
        feedback.append("Boa! Atividade física faz bem ao humor.")

    return feedback


# -------------------------------------------------------------
# INTERFACE
# -------------------------------------------------------------
st.title("🧠 Bem-Estar IA")
st.write("Aplicativo de análise emocional e hábitos usando Deep Learning.")

tabs = st.tabs(["📸 Análise de Emoções", "📆 Avaliação de Rotina"])


# =============================================================
# ABA 1 — ANÁLISE DE EMOÇÕES
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

                st.markdown("### 📊 Probabilidades:")
                for label, p in zip(emotion_labels, probs):
                    st.write(f"- **{label}** → {p*100:.2f}%")

                st.markdown("### 💡 Recomendações:")
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
        st.markdown("### 📋 Resultados:")

        feedback = analisar_rotina(sono, trabalho, lazer, exercicio)

        for f in feedback:
            st.markdown(f"- {f}")

        st.markdown("### ✨ Dica Final:")
        st.write("Tente registrar sua rotina diariamente para acompanhar sua evolução.")

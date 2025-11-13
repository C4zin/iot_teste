import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

# ---------------- CONFIGURAÇÃO GERAL ----------------
st.set_page_config(
    page_title="Bem-Estar IA",
    page_icon="🧠",
    layout="wide"
)

# ---------------- FUNÇÕES AUXILIARES ----------------
def traduz_emocao(emocao_en):
    mapa = {
        "happy": "feliz",
        "sad": "triste",
        "angry": "irritado(a)",
        "fear": "com medo/ansioso(a)",
        "surprise": "surpreso(a)",
        "neutral": "neutro(a)",
        "disgust": "desgostoso(a)"
    }
    return mapa.get(emocao_en, emocao_en)

def sugestoes_por_emocao(emocao_en):
    sugestoes = {
        "happy": [
            "Mantenha hábitos que estão funcionando: sono razoável, pausas e lazer.",
            "Aproveite o bom momento para iniciar um novo hábito saudável (ex.: caminhada diária).",
            "Compartilhe algo positivo com alguém — isso reforça seu bem-estar."
        ],
        "sad": [
            "Tente organizar seu dia reservando um tempo fixo para lazer ou algo que você goste.",
            "Evite longos períodos sem pausas: levante, alongue-se, tome água a cada 60–90 minutos.",
            "Se a tristeza for frequente, considere conversar com um profissional de saúde mental."
        ],
        "angry": [
            "Inclua na rotina pequenas pausas de respiração profunda quando estiver irritado.",
            "Planeje os momentos críticos do dia (reuniões, provas, trânsito) com folga de horário.",
            "Atividades físicas regulares ajudam a reduzir tensão e irritabilidade."
        ],
        "fear": [
            "Liste as principais preocupações do dia e defina pequenas ações para cada uma.",
            "Evite uso excessivo de telas próximo ao horário de dormir.",
            "Inclua na rotina uma atividade relaxante (meditação guiada, leitura leve, música)."
        ],
        "surprise": [
            "Revise sua agenda para evitar imprevistos recorrentes.",
            "Use um bloco de notas ou app para registrar compromissos importantes.",
            "Mantenha horários fixos para refeições e sono, reduzindo impactos de surpresas."
        ],
        "neutral": [
            "Experimente inserir um pequeno momento de lazer obrigatório no dia.",
            "Defina uma meta simples para hoje (ex.: 10 min de alongamento).",
            "Avalie como foi seu sono e alimentação: pequenos ajustes geram grande impacto."
        ],
        "disgust": [
            "Identifique atividades que geram mais desconforto e tente distribuí-las ao longo da semana.",
            "Inclua algo prazeroso logo após tarefas desagradáveis como recompensa.",
            "Reflita se não há excesso de obrigações; renegociar prazos quando possível é saudável."
        ]
    }
    return sugestoes.get(emocao_en, ["Cuide de você, mantenha uma rotina equilibrada."])

def analise_rotina(horas_sono, horas_trabalho, horas_lazer, horas_exercicio):
    feedback = []

    if horas_sono < 7:
        feedback.append("Você está dormindo pouco. Tente se aproximar de 7–8h de sono por noite.")
    elif horas_sono > 9:
        feedback.append("Você está dormindo bastante. Veja se isso não está ligado à fadiga ou desânimo.")
    else:
        feedback.append("Seu tempo de sono está em uma faixa saudável. Mantenha esse hábito! 😴")

    if horas_trabalho > 9:
        feedback.append("Muitas horas de trabalho/estudo. Tente inserir pausas e definir limites claros.")
    elif horas_trabalho < 4:
        feedback.append("Poucas horas produtivas. Talvez definir blocos de foco ajude na organização.")
    else:
        feedback.append("Carga de trabalho/estudo equilibrada. Continue organizando bem seu dia. 📚")

    if horas_lazer < 1:
        feedback.append("Quase sem lazer. Separe pelo menos 30–60 min para algo que você goste todos os dias.")
    else:
        feedback.append("Bom ver que você tem um tempo para lazer. Isso ajuda muito na saúde mental. 🎮📖")

    if horas_exercicio == 0:
        feedback.append("Tente incluir ao menos 10–20 min de caminhada ou alongamento no dia.")
    elif horas_exercicio < 3:
        feedback.append("Você faz um pouco de exercício. Que tal aumentar gradualmente a frequência?")
    else:
        feedback.append("Excelente! Sua rotina de exercícios é um ponto muito positivo para o bem-estar. 🏃‍♀️")

    return feedback

# ---------------- LAYOUT ----------------
st.title("🧠 Bem-Estar IA")
st.markdown("""
Aplicativo baseado em **Deep Learning** para auxiliar na organização e melhoria da sua rotina diária.

- Análise inteligente das **emoções faciais** (Visão Computacional).
- Registro de **hábitos diários** (sono, trabalho/estudo, lazer, exercícios).
- Sugestões personalizadas para promover **bem-estar físico e mental**.
""")

tab1, tab2 = st.tabs(["📸 Análise de Emoções (Deep Learning)", "📆 Hábitos e Rotina"])

# ---------------- TAB 1: EMOÇÕES ----------------
with tab1:
    st.subheader("Envie uma foto do seu rosto")

    st.write("A imagem será analisada por um modelo de Deep Learning (biblioteca **DeepFace**).")

    arquivo = st.file_uploader(
        "Escolha uma foto (formatos: JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if arquivo is not None:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, caption="Foto enviada", use_container_width=True)

        if st.button("Analisar emoção"):
            with st.spinner("Analisando emoções com Deep Learning..."):
                try:
                    resultado = DeepFace.analyze(
                        np.array(imagem),
                        actions=["emotion"],
                        enforce_detection=True
                    )
                    # DeepFace pode retornar lista em algumas versões
                    if isinstance(resultado, list):
                        resultado = resultado[0]

                    emocao_dom = resultado.get("dominant_emotion", "neutral")
                    emocao_pt = traduz_emocao(emocao_dom)

                    st.success(f"Emoção predominante detectada: **{emocao_pt}**")

                    st.markdown("### Sugestões com base na sua emoção atual:")
                    for s in sugestoes_por_emocao(emocao_dom):
                        st.markdown(f"- {s}")

                except Exception as e:
                    st.error(
                        "Não foi possível detectar um rosto com clareza na imagem. "
                        "Tente outra foto com boa iluminação e o rosto voltado para a câmera."
                    )
                    st.caption(f"Detalhes técnicos: {e}")

# ---------------- TAB 2: ROTINA ----------------
with tab2:
    st.subheader("Como está sua rotina hoje?")

    col1, col2 = st.columns(2)

    with col1:
        horas_sono = st.slider("Horas de sono por noite", 0.0, 12.0, 7.0, 0.5)
        horas_trabalho = st.slider("Horas de trabalho/estudo por dia", 0.0, 14.0, 8.0, 0.5)

    with col2:
        horas_lazer = st.slider("Horas de lazer por dia", 0.0, 8.0, 1.0, 0.5)
        horas_exercicio = st.slider("Horas de exercício físico por dia", 0.0, 4.0, 0.0, 0.5)

    if st.button("Gerar análise da rotina"):
        st.markdown("### Análise dos seus hábitos de hoje:")
        feedbacks = analise_rotina(horas_sono, horas_trabalho, horas_lazer, horas_exercicio)

        for f in feedbacks:
            st.markdown(f"- {f}")

        st.markdown("#### Dica extra")
        st.write(
            "Tente registrar sua rotina diariamente. Com o tempo, você pode acompanhar a evolução "
            "dos seus hábitos e perceber como pequenas mudanças impactam seu bem-estar."
        )

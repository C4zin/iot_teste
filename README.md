
# 🧠 Bem-Estar IA
Aplicação web baseada em **Deep Learning e Visão Computacional** para análise emocional e avaliação de rotina, com foco em promover **bem-estar mental e equilíbrio diário**.  
O projeto utiliza uma **rede neural convolucional (CNN)** para classificar emoções faciais e integrar esses resultados a um sistema de recomendações personalizadas.

---

## 🎯 Objetivo do Projeto
O Bem-Estar IA foi desenvolvido para:

- Analisar a **emoção do usuário** a partir de uma imagem facial  
- Avaliar a **rotina diária** (sono, trabalho, lazer e exercícios)  
- Oferecer **sugestões personalizadas** de bem-estar  
- Integrar Deep Learning, visão computacional e análise de hábitos em uma aplicação moderna e intuitiva  

---

## 🧪 Tecnologias Utilizadas

### Back-end / IA
- Python 3.10  
- TensorFlow 2.x  
- Keras  
- NumPy  
- OpenCV  
- Pillow  

### Aplicação Web
- Streamlit Cloud  

### Inteligência Artificial
- CNN criada no código  
- Entradas 48×48 grayscale  
- Classes emocionais: Raiva, Nojo, Medo, Feliz, Triste, Surpreso, Neutro  

---

## 📂 Arquitetura do Projeto

```
Bem-Estar-IA/
│── app.py               # Aplicação principal (Streamlit)
│── requirements.txt     # Dependências
│── emotion_tf2.h5       # Modelo criado 
│── README.md            # Documentação
```

---

## ⚙️ Como o Projeto Funciona

### 1️⃣ Processamento da Imagem  
Converte a imagem enviada para grayscale 48×48 e envia para a CNN.

### 2️⃣ Predição da Emoção  
O modelo retorna probabilidades para as 7 emoções.

### 3️⃣ Avaliação de Rotina  
O usuário informa:
- horas de sono  
- trabalho/estudo  
- lazer  
- exercícios  

### 4️⃣ Recomendações  
O sistema combina emoção + rotina e gera dicas personalizadas.

---

## 🚀 Como Executar Localmente

### Clone o repositório
```
git clone https://github.com/seuprojeto/Bem-Estar-IA.git
cd Bem-Estar-IA
```

### Instale dependências
```
pip install -r requirements.txt
```

### Execute o app
```
streamlit run app.py
```

---

## ☁️ Deploy no Streamlit Cloud

1. Envie o projeto ao GitHub  
2. Vá até https://share.streamlit.io  
3. Clique em Deploy App  
4. Selecione o repositório  
5. O Streamlit encontrará:
   - `app.py`
   - `requirements.txt`

O modelo é criado automaticamente.

---

## 🧩 Integração com Disciplinas

- Visão Computacional  
- Deep Learning  
- Desenvolvimento Web  
- Saúde e Bem-estar  
- HCI/UX  

---

## 📚 Critérios de Avaliação — Atendidos

| Critério | Status |
|---------|--------|
| Deep Learning (60 pts) | ✔ CNN funcional |
| Integração interdisciplinar (20 pts) | ✔ IA + saúde + rotina |
| Boas práticas (10 pts) | ✔ Código modular e limpo |
| Apresentação (10 pts) | ✔ Interface de fácil demonstração |

---

## 👨‍💻 Autor
Seu Nome  
Projeto acadêmico de Inteligência Artificial e Visão Computacional.

---

## 🤝 Contribuições
Sinta-se livre para enviar melhorias ou sugestões.

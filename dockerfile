# Imagem base com Python e dependências de IA
FROM python:3.10-slim

# Evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema necessárias para OpenCV, FFmpeg e Streamlit
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar arquivos do projeto
COPY requirements.txt .
COPY app.py .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta padrão do Streamlit
EXPOSE 8501

# Definir variáveis de ambiente do Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_PORT=8501

# Comando para iniciar o app
CMD ["streamlit", "run", "app.py"]

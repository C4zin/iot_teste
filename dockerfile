# Imagem base com Python 3.10
FROM python:3.10-slim

# Evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema necessárias para OpenCV, ffmpeg e Streamlit
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar os arquivos do projeto
COPY requirements.txt .
COPY app.py .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta padrão do Streamlit
EXPOSE 8501

# Variáveis do Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_PORT=8501

# Comando para iniciar o app
CMD ["streamlit", "run", "app.py"]

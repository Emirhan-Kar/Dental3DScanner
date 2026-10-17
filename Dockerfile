# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Temel ortam değişkenleri (etkileşimsiz ve headless görselleştirme)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    QT_QPA_PLATFORM=offscreen

WORKDIR /app

# OpenCV ve Open3D için gerekli sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libusb-1.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Proje kaynak kodlarını kopyala
COPY . .

# Çıktı ve veri dizinlerinin varlığını garantiye al
RUN mkdir -p output logs data

# Varsayılan çalıştırma komutu (sentetik simülasyon modu)
CMD ["python", "main.py", "--simulation", "--output", "output/scan_result.ply", "--export-mesh", "output/dental_model.stl"]

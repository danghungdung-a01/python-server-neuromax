FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p storage

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Quan trọng: chỉ 1 worker, timeout dài hơn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "app:app"]

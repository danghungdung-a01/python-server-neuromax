# ==============
# Stage 1: Base Image
# ==============
FROM python:3.11-slim

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ==============
# Stage 2: Application setup
# ==============
WORKDIR /app

# Copy Python dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Create storage directory (avoid errors)
RUN mkdir -p storage

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Expose Flask port
EXPOSE 5000

# Start the app using Gunicorn (recommended for production)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

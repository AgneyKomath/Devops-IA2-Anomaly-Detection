FROM python:3.11-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential tzdata ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Streamlit runtime defaults
ENV PYTHONUNBUFFERED=1 \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

# Ensure dirs exist (in case repo is fresh)
RUN mkdir -p models data/raw

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]

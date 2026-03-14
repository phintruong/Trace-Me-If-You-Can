# Fraud backend: FastAPI + pipeline. Port 8080.
# Build from repo root: docker build -f backend/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application
COPY backend/app ./app
COPY backend/run_pipeline.py ./
COPY backend/model ./model

# Copy datasets (if present) so pipeline can load data inside container
COPY datasets ./datasets

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

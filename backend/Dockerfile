# Fraud backend: FastAPI + pipeline. Port 8080.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (build from repo root: docker build -f backend/Dockerfile .)
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app and project src (for pipeline loader/features)
COPY backend/app ./app
COPY backend/run_pipeline.py ./
COPY src ./src

# Ensure app can import app.* and src.*
ENV PYTHONPATH=/app

EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

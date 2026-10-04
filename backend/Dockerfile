FROM python:3.10-slim
WORKDIR /app
COPY requirements-backend.txt /app/requirements-backend.txt
RUN pip install --no-cache-dir -r requirements-backend.txt
COPY src /app/src
COPY backend /app/backend
ENV PYTHONPATH=/app/src MODEL_DIR=/models PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


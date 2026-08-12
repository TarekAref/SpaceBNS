FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN python -m pip install --no-cache-dir -r /workspace/backend/requirements.txt

COPY backend /workspace/backend
COPY data /workspace/data

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


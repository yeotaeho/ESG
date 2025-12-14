FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn

COPY gateway ./gateway
COPY service ./service

CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "9000"]

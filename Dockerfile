FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt update && apt install -y build-essential

COPY . .

ARG EXTRAS=openai,ollama
RUN pip install --prefix=/install --no-cache-dir .[$EXTRAS]

# ---- Final Image ----
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

CMD ["python", "src/emanuel/main.py"]
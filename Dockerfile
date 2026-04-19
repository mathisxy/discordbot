FROM python:3.13-slim

WORKDIR /app

COPY . .

ARG EXTRAS=openai,ollama
RUN pip install --no-cache-dir .[$EXTRAS]

CMD ["python", "src/emanuel/main.py"]
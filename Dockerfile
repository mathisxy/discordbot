FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY . .

ARG EXTRAS=openai,ollama
RUN pip install --no-cache-dir .[$EXTRAS]

CMD ["python", "src/emanuel/main.py"]
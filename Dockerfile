FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .[openai,ollama]

CMD ["python", "main.py"]
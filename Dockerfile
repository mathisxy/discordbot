FROM python:3.13-slim

WORKDIR /app

COPY . .

# TODO: Integrate in github actions
ARG EXTRAS=openai,ollama

RUN pip install --no-cache-dir .[$EXTRAS]

CMD ["python", "src/emanuel/main.py"]
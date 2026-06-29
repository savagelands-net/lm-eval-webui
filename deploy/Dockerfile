# syntax=docker/dockerfile:1
FROM docker:27-cli AS docker-cli

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.bun/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

RUN curl -fsSL https://bun.sh/install | bash

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "-m", "lm_eval_webui", "--host", "0.0.0.0", "--port", "8080", "--data-dir", "/data", "--static-dir", "/app/static", "--pi-bench-dir", "/app/third_party/pi-bench"]

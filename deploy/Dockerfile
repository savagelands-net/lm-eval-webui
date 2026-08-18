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

# BuildKit supplies TARGETARCH per target platform; a default would override arm64.
ARG TARGETARCH
ARG LEMONADE_CLI_VERSION=11.6.0
ARG LEMONADE_CLI_SHA256_AMD64=9d3fc9c96e745e96c56347eb0a9c1b07ad791fbd8def47beb9fe3ef5fc40e8de
ARG LEMONADE_CLI_SHA256_ARM64=69c3a57fd9ed16439d977543fc10ecf486f86e55d04aa9b46626464d6e99a6b4
RUN case "${TARGETARCH}" in \
        amd64) lemonade_arch=x64; lemonade_sha="${LEMONADE_CLI_SHA256_AMD64}" ;; \
        arm64) lemonade_arch=arm64; lemonade_sha="${LEMONADE_CLI_SHA256_ARM64}" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL \
        "https://github.com/lemonade-sdk/lemonade/releases/download/v${LEMONADE_CLI_VERSION}/lemonade-embeddable-${LEMONADE_CLI_VERSION}-ubuntu-${lemonade_arch}.tar.gz" \
        -o /tmp/lemonade-cli.tar.gz \
    && echo "${lemonade_sha}  /tmp/lemonade-cli.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/lemonade \
    && tar -xzf /tmp/lemonade-cli.tar.gz -C /opt/lemonade --strip-components=1 \
    && ln -s /opt/lemonade/lemonade /usr/local/bin/lemonade \
    && lemonade --version \
    && rm /tmp/lemonade-cli.tar.gz

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
# pi-lens-ignore: dockerfile.security.missing-user.missing-user, opengrep:dockerfile.security.missing-user.missing-user
CMD ["python", "-m", "lm_eval_webui", "--host", "0.0.0.0", "--port", "8080", "--data-dir", "/data", "--static-dir", "/app/static", "--max-request-workers", "16", "--pi-bench-dir", "/app/third_party/pi-bench"]

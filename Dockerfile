# EDM-ARS Code Sandbox — deterministic execution environment
# Build: docker build -t edm-ars-sandbox:latest .
# Used ONLY for executing LLM-generated analysis code.

FROM python:3.11.9-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r sandbox && useradd -r -g sandbox -m -d /home/sandbox sandbox

COPY requirements-sandbox.txt /tmp/requirements-sandbox.txt
RUN pip install --no-cache-dir -r /tmp/requirements-sandbox.txt && \
    rm /tmp/requirements-sandbox.txt

RUN python -c "import pandas, numpy, sklearn, xgboost, shap, matplotlib; print('All packages OK')"

ENV MPLBACKEND=Agg

WORKDIR /workspace

USER sandbox

ENTRYPOINT ["python", "-c"]

FROM python:3.11-slim-bookworm

# Dependencies stay in the container. No GPU or display is needed for FK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglfw3 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace
CMD ["python", "docs/reports/act-sonic-joint-pose-2026-09-07/reproduce.py"]

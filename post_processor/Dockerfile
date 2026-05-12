# Use a Python base image with Debian for system dependencies
FROM python:3.11-slim-bookworm

WORKDIR /post_processor

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    clang \ 
    llvm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install  --no-cache-dir -r requirements.txt 

RUN git clone --recursive https://github.com/microsoft/BitNet.git /BitNet

RUN chmod -R a+rX /BitNet && \
    chmod +x /BitNet/run_inference.py || true

RUN pip install --no-cache-dir -r /BitNet/requirements.txt

RUN mkdir -p /post_processor/models && \
    huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir /post_processor/models/BitNet-b1.58-2B-4T

RUN cd /BitNet && \
    python setup_env.py -md /post_processor/models/BitNet-b1.58-2B-4T -q i2_s


COPY post_processor.py /app/post_processor.py

CMD ["python", "-u", "/app/post_processor.py"]
FROM nvidia-llm-dev
#nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-runtime-ubuntu24.04

LABEL maintainer="kaikai.liu@sjsu.edu"
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV CMAKE_PREFIX_PATH=/usr/local/cuda

# Expose ports for JupyterLab, Ollama, llama.cpp server
EXPOSE 8888 11434 8000

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    curl \
    wget \
    python3 python3-pip python3-venv \
    python-is-python3 \
    ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Setup venv and install dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Create a script to activate the virtual environment
RUN echo '#!/bin/bash\n\
source /opt/venv/bin/activate\n\
exec "$@"' > /usr/local/bin/with_venv && \
    chmod +x /usr/local/bin/with_venv

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install setuptools wheel "vllm"

# Python packages: core + ML/NLP/LLM
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    tqdm \
    huggingface_hub \
    transformers \
    sentence-transformers \
    langchain \
    tabulate \
    requests \
    fastapi \
    uvicorn \
    jupyterlab

# Clone and build llama.cpp with CUDA/cuBLAS
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev
RUN ln -s /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so && \
    ln -s /usr/local/cuda/lib64/libcublasLt.so.12 /usr/local/cuda/lib64/libcublasLt.so
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so || true
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 || true
RUN find /opt/venv/lib/python*/site-packages/nvidia/cublas/include -name '*.h' \
  -exec ln -sf {} /usr/local/cuda/include/ \;
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp
RUN mkdir build && cd build && cmake .. -DGGML_CUDA=ON && cmake --build . --config Release

# Install llama-cpp-python with CUDA/cuBLAS
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install Ollama (if available for ARM64 — fallback warning)
RUN mkdir -p /opt/ollama && cd /opt/ollama && \
    curl -L https://ollama.com/download/ollama-linux-arm64 -o ollama && \
    chmod +x ollama && mv ollama /usr/local/bin/ollama || echo "⚠️ Ollama ARM64 not available — skip"

RUN apt-get update && apt-get install -y x11-apps

# Create shared volume for models
RUN mkdir -p /models /root/.cache

# Default working directory
WORKDIR /workspace

# Set the entrypoint to use our virtual environment activation script
ENTRYPOINT ["/usr/local/bin/with_venv"]

# Default startup command: launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

#run this build: docker build -t nvidia-llm-dev .

#Run the Container with Ports and Volumes
# docker run --rm -it --runtime nvidia \
#   -p 8888:8888 -p 11434:11434 -p 8000:8000 \
#   -v $(pwd)/models:/models \
#   -v $(pwd)/workspace:/workspace \
#   jetson-llm-lab
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip \
    gcc g++ git ninja-build \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/include/crypt.h /usr/local/include/crypt.h 2>/dev/null || true

WORKDIR /app
COPY . .

# Pin numpy FIRST before anything else
RUN pip3 install "numpy<2" --force-reinstall
RUN pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# Force numpy again because torch install may upgrade it
RUN pip3 install "numpy<2" --force-reinstall
RUN pip3 install -r requirements.txt
RUN pip3 install fastapi uvicorn python-multipart
# Final numpy pin
RUN pip3 install "numpy<2" --force-reinstall

# Build CUDA extensions with correct numpy
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV LDFLAGS="-L/lib/x86_64-linux-gnu"
RUN python -c "import numpy; print('NumPy version before build:', numpy.__version__)"
RUN cd lib && python setup.py build develop

EXPOSE 8000
CMD ["python", "hand_detect_api.py"]

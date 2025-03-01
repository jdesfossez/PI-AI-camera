#FROM tensorflow/tensorflow:2.14.0-gpu # need Python < 3.11, this one has 3.11.0rc1
#FROM python:3.10-slim

FROM nvidia/cuda:12.4.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    unzip \
    curl \
    libgl1-mesa-glx \
    nano \
    wget \
    tree \
    openjdk-17-jdk \
    ffmpeg libsm6 libxext6 \
    pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#RUN cat $(python3 --version)

RUN pip install -U pip && \
    pip install --no-cache-dir \
        tensorflow~=2.14.0 \
        torch~=1.13.1 \
        torchvision==0.14.1 \
        imx500-converter[tf] \
        pycocotools

WORKDIR /root

RUN git clone https://github.com/RangiLyu/nanodet.git && \
    touch nanodet/nanodet/model/__init__.py && \
    cd nanodet && \
    git checkout be9b4a9 && \
    pip install -q --no-cache-dir -r requirements.txt

RUN git clone https://github.com/sony/model_optimization.git local_mct && \
    cd local_mct && \
    git checkout v2.1.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    cd ..

ENV PATH="$PATH:/root/local_mct"
#ENV PYTHONPATH=/root/nanodet:$PYTHONPATH

COPY train-det.py .
COPY nanodet-plus-m-1.5x_416-ppe.yml .

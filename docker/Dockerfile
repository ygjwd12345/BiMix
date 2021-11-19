# set base image (host OS)
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0"

RUN apt-get clean && apt-get update && apt-get install -y \
    tree \
    zip \
    vim \
    python3-tk \
    git \
    unzip \
    htop \
    wget \
    libgl1-mesa-dev\
    libgtk2.0-dev\
    tmux\
    screen
RUN pip install --no-cache-dir Cython \
    wandb \
    mmcv \
    einops \
    torch-encoding \
    torchvision \
    ninja \
    scikit-image \
    opencv-python \
    pandas \
    tqdm \
    scikit-learn \
    inplace_abn \
    screen \
    tensorboardX \
    scipy

RUN pip install pycocotools \
    opencv-contrib-python

WORKDIR "/home"
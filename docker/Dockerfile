ARG CUDA_VERSION=12.8.1

# mmcv 2.x has no prebuilt wheels for torch 2.7 / CUDA 12.8, so its ops are compiled here
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS mmcv-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir setuptools==69.5.1 wheel "numpy<2"
RUN python3 -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# same GPU architectures as the torch 2.7.1+cu128 wheel, Blackwell (10.0, 12.0) included
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX" \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    MAX_JOBS=4
RUN git clone --depth 1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git /tmp/mmcv \
    && cd /tmp/mmcv \
    && python3 -m pip wheel --no-build-isolation --no-deps -w /tmp/wheels .


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgeos-dev \
    libmagic-dev \
    libexiv2-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# torch >= 2.6 loads checkpoints with weights_only=True, which mmengine checkpoints do not pass
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# keep numpy 1.x (and an opencv that supports it) for mmcv/mmdet
RUN printf 'numpy<2\nopencv-python<4.12\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN pip install --no-cache-dir setuptools==69.5.1 "numpy<2"
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir mmengine==0.10.7 yapf==0.40.1

COPY --from=mmcv-builder /tmp/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/mmcv-*.whl && rm -rf /tmp/wheels

# mmdet
RUN pip install --no-cache-dir mmdet==3.3.0 mmcls==1.0.0rc6
# mmcls (ConvNeXt backbones) caps mmcv below 2.1.0, the first mmcv that builds with torch 2.7
RUN sed -i "s/^mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.2.0'/" \
    /usr/local/lib/python3.10/dist-packages/mmcls/__init__.py

RUN pip install --no-cache-dir supervisely==6.73.486
RUN pip install --no-cache-dir supervisely[aug]==6.73.486
RUN pip install --no-cache-dir supervisely[model-benchmark]==6.73.486
RUN pip install --no-cache-dir supervisely[tracking]==6.73.486

RUN python -c "import torch, mmcv, mmdet, mmcls, imgaug; from mmcv.ops import nms; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmdet.__version__)"

LABEL python_sdk_version="6.73.486"

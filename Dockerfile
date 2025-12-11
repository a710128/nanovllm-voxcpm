# VoxCPM TTS Docker Image
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends     wget git ca-certificates ninja-build     && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh &&     bash /tmp/miniconda.sh -b -p /opt/conda &&     rm /tmp/miniconda.sh &&     /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# 接受 Conda TOS 并创建环境
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &&     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r &&     conda create -n voxcpm python=3.11 -y

# 安装 PyTorch 2.4.0 (固定版本)
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir     torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir     -i https://mirrors.aliyun.com/pypi/simple/     fastapi uvicorn pydantic soundfile einops transformers aiohttp packaging

# 安装 flash-attn
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=4
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir --no-build-isolation     -i https://mirrors.aliyun.com/pypi/simple/     flash-attn

# 复制代码和模型
COPY nanovllm-voxcpm /app/nanovllm-voxcpm
COPY VoxCPM-0.5B /app/VoxCPM-0.5B
COPY frontend /app/frontend

# 安装项目
WORKDIR /app/nanovllm-voxcpm
RUN /opt/conda/envs/voxcpm/bin/pip install -e .

EXPOSE 8081

CMD ["/opt/conda/envs/voxcpm/bin/uvicorn", "api_server.app:app", "--host", "0.0.0.0", "--port", "8081"]

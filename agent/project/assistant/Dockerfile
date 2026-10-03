# ============================================================
#  个人知识库助手 —— Docker 镜像
#
#  构建上下文 = 本目录，所以在本目录执行：
#     docker build -t kb-assistant .
#     docker run -p 8000:8000 -e DEEPSEEK_API_KEY=sk-xxx kb-assistant
# ============================================================
# ★ 基础镜像钉住 digest = 可复现构建。
#
#   为什么必须钉：python:3.10-slim 是【可变标签】，它指向的 digest 会随时间变化。
#   而 Docker 的分层缓存是【按父层串行】的 —— 父层一变，后面每一层全部失效。
#   实测：Docker 引擎重启后标签被重新解析、拿到了新 digest，
#         结果 COPY requirements.txt / pip install torch 全部重跑，
#         构建从 ~1 分钟变成 5.5 分钟（重下 196 MB 的 torch）。
#
#   代价：不再自动获得基础镜像的安全更新 → 需要定期手动升级下面这个 digest。
#   （可复现 和 自动更新 不能同时要，得用流程补：Renovate / Dependabot 定期提 PR）
#
#   查询 digest（Docker Hub 直连被墙时走镜像站）：
#     docker buildx imagetools inspect docker.m.daocloud.io/library/python:3.10-slim `
#         --format "{{.Manifest.Digest}}"
FROM python:3.10-slim@sha256:c1aaf3d03e14944a039a1647e0b3f6f34c6bee517bac6ff380215ee099c4e808

# ★ PYTHONUNBUFFERED=1 很关键：
#   Python 默认缓冲 stdout，容器里 print 的内容会被攒着，
#   docker logs 看不到实时输出（启动日志、报错都延迟）。关掉缓冲。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_ENDPOINT=https://hf-mirror.com

# 依赖源。可在构建时覆盖，例如：
#   docker build --build-arg PIP_INDEX=https://mirrors.aliyun.com/pypi/simple -t kb-assistant .
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

WORKDIR /app

# ---- 第 1 层：依赖 ----------------------------------------------------
# 先只 COPY requirements.txt 再装依赖：
# 只要它没变，改代码时这一层命中缓存，不用重装（省几分钟）
COPY requirements.txt .

# ★ 故意拆成三个 RUN，而不是一条 && 串到底：
#   任何一步失败，前面成功的层仍然留在缓存里，重试时不用重下
RUN pip install --upgrade pip -i ${PIP_INDEX}

# ★ torch 必须先单独装 CPU 版：
#   否则从 PyPI 装会拉 CUDA 版（2GB+），我们 0 显存用不上
RUN pip install torch --index-url ${TORCH_INDEX}

RUN pip install -r requirements.txt -i ${PIP_INDEX}

# ---- 第 2 层：代码 ----------------------------------------------------
COPY . .

EXPOSE 8000

# ★ --host 0.0.0.0 必须写死：
#   uvicorn 默认只监听 127.0.0.1，那是"容器内部"的回环地址，
#   宿主机/外网根本连不上。容器里的服务一律监听 0.0.0.0。
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
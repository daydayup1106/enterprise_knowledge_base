# 企业知识库专家协同系统 — 腾讯云部署指南

> **适用环境**：腾讯云 CVM（Ubuntu 22.04 LTS）+ Docker Compose  
> **部署栈**：FastAPI + Redis Stack + 本地 Embedding/Reranker 模型

---

## 一、服务器最低配置要求

| 资源 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 | 8 核 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 40 GB SSD | 100 GB SSD |
| GPU | 无（CPU推理，较慢） | 1× NVIDIA GPU（4GB+ 显存） |
| 带宽 | 5 Mbps | 10 Mbps |
| 操作系统 | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

> **说明**：Embedding 模型（bge-small-zh-v1.5，约 100MB）和 Reranker 模型（bge-reranker-base，约 1GB）在 CPU 上可正常运行，但首次请求推理耗时较长（5～15秒），有 GPU 时性能大幅提升。

---

## 二、腾讯云安全组配置

登录腾讯云控制台 → 云服务器 → 安全组 → 入站规则，**添加以下规则**：

| 协议 | 端口 | 来源 | 用途 |
|------|------|------|------|
| TCP | 22 | 你的本机IP | SSH 登录 |
| TCP | 8181 | 0.0.0.0/0 | FastAPI 应用端口 |
| TCP | 6379 | 127.0.0.1/32 | Redis（仅本机访问，**不要对公网开放**） |

---

## 三、服务器初始化

### 3.1 SSH 登录服务器

```bash
ssh ubuntu@<你的腾讯云公网IP>
```

### 3.2 安装 Docker 和 Docker Compose

```bash
# 更新包索引
sudo apt-get update && sudo apt-get upgrade -y

# 安装 Docker（官方一键脚本）
curl -fsSL https://get.docker.com | sudo bash

# 将当前用户加入 docker 组（免 sudo）
sudo usermod -aG docker $USER
newgrp docker

# 验证安装
docker --version
docker compose version
```

### 3.3 （可选）配置 Docker 镜像加速

在国内拉取 Docker Hub 镜像较慢，建议配置腾讯云镜像源：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com"
  ]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

---

## 四、上传项目代码

### 方式 A：Git 克隆（推荐）

```bash
# 在服务器上
sudo apt-get install git -y
git clone <你的代码仓库地址> /opt/enterprise_knowledge_base
cd /opt/enterprise_knowledge_base
```

### 方式 B：本地 SCP 上传

```bash
# 在本机执行（Windows 可用 PowerShell 或 WinSCP）
scp -r d:\pycharm_workspace\enterprise_knowledge_base ubuntu@<公网IP>:/opt/enterprise_knowledge_base
```

---

## 五、下载本地模型权重

本项目使用两个 HuggingFace 模型，需要提前下载到服务器。

```bash
# 在服务器上安装 huggingface_hub（临时用）
pip3 install huggingface_hub -q

# 创建模型目录
mkdir -p /opt/models

# 下载 Embedding 模型（约 100MB）
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='/opt/models/bge-small-zh-v1.5')
"

# 下载 Reranker 模型（约 1GB）
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-base', local_dir='/opt/models/bge-reranker-base')
"
```

> **网络问题**：若 HuggingFace 下载慢，可使用镜像站：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```
> 然后再执行上面的下载命令。

也可以**从本机上传已下载的模型**（更可靠）：
```bash
# 在本机执行
scp -r D:\models\bge-small-zh-v1.5 ubuntu@<公网IP>:/opt/models/bge-small-zh-v1.5
scp -r D:\models\bge-reranker-base ubuntu@<公网IP>:/opt/models/bge-reranker-base
```

---

## 六、配置环境变量

```bash
cd /opt/enterprise_knowledge_base

# 从模板复制 .env 文件
cp .env.example .env

# 编辑 .env
nano .env
```

填写以下内容（其余保持默认即可）：

```dotenv
# ---- DeepSeek API ----
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx   # 替换为你的真实 Key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# ---- 本地模型路径（服务器上的绝对路径）----
EMBED_MODEL_NAME=/opt/models/bge-small-zh-v1.5
RERANK_MODEL_NAME=/opt/models/bge-reranker-base

# CPU 推理（无 GPU 时）；有 GPU 时改为 cuda
DEVICE=cpu

# ---- Redis（docker-compose 内通信，不要修改）----
REDIS_URL=redis://redis:6379/0
```

> ⚠️ **安全提示**：`.env` 文件包含 API Key，确保其不被提交到 Git 仓库（`.gitignore` 已配置）。

---

## 七、修复 docker-compose.yml 模型挂载路径

当前 `docker-compose.yml` 中模型 volume 写法在 Linux 服务器上需要调整——Docker Compose 
无法直接将环境变量作为宿主机目录路径来挂载，需改为**具体的绝对路径**：

打开 `docker-compose.yml`，将 `web` 服务的 `volumes` 部分修改为：

```yaml
    volumes:
      - /opt/models/bge-small-zh-v1.5:/app/models/embed_model:ro
      - /opt/models/bge-reranker-base:/app/models/rerank_model:ro
      - ./data:/app/data:ro
```

同时，在 `environment` 中将模型路径改为容器内路径：

```yaml
    environment:
      - EMBED_MODEL_NAME=/app/models/embed_model
      - RERANK_MODEL_NAME=/app/models/rerank_model
```

修改后的完整 `docker-compose.yml` 示例见[第十一章附录](#十一附录完整-docker-composeyml)。

---

## 八、构建并启动服务

```bash
cd /opt/enterprise_knowledge_base

# 首次构建（会下载基础镜像并安装依赖，需 10～20 分钟）
docker compose build

# 后台启动所有服务
docker compose up -d

# 查看启动日志（确认无报错）
docker compose logs -f web
```

**正常启动的日志应包含**：

```
System READY — XX knowledge chunks indexed.
```

---

## 九、验证部署

```bash
# 健康检查接口
curl http://localhost:8181/api/v1/health

# 期望返回
# {"status":"ready","index_loaded":true,"doc_chunk_count":XX}
```

在浏览器访问：`http://<你的腾讯云公网IP>:8181`

---

## 十、日常运维命令

```bash
# 查看所有容器状态
docker compose ps

# 查看实时日志
docker compose logs -f web
docker compose logs -f redis

# 重启 web 服务（修改代码/配置后）
docker compose restart web

# 完全停止并销毁容器（数据 volume 保留）
docker compose down

# 完全销毁包括数据卷（慎用，会丢失 Redis 对话历史）
docker compose down -v

# 更新代码后重新构建并启动
git pull
docker compose build web
docker compose up -d web

# 进入容器调试
docker exec -it ekb_web bash
```

---

## 十一、附录：完整 docker-compose.yml

将当前 `docker-compose.yml` 替换为以下内容（已修复模型挂载路径问题）：

```yaml
version: '3.8'

services:
  web:
    build: .
    container_name: ekb_web
    ports:
      - "8181:8181"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}
      - DEEPSEEK_MODEL=${DEEPSEEK_MODEL}
      # 容器内固定路径，通过 volumes 挂载宿主机模型目录
      - EMBED_MODEL_NAME=/app/models/embed_model
      - RERANK_MODEL_NAME=/app/models/rerank_model
      - DEVICE=${DEVICE:-cpu}
      - RAG_HYBRID_TOP_K=${RAG_HYBRID_TOP_K:-10}
      - RAG_RERANK_TOP_N=${RAG_RERANK_TOP_N:-3}
      - CHUNK_SIZE=${CHUNK_SIZE:-512}
      - CHUNK_OVERLAP=${CHUNK_OVERLAP:-128}
      - MEMORY_WINDOW=${MEMORY_WINDOW:-10}
      - REDIS_URL=redis://redis:6379/0
      - KNOWLEDGE_BASE_DOC=${KNOWLEDGE_BASE_DOC:-data/medical_ai_papers_2024_2025.md}
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      # 将宿主机上已下载的模型目录挂载到容器（只读）
      - /opt/models/bge-small-zh-v1.5:/app/models/embed_model:ro
      - /opt/models/bge-reranker-base:/app/models/rerank_model:ro
      - ./data:/app/data:ro
    restart: unless-stopped

  redis:
    image: redis/redis-stack-server:latest
    container_name: ekb_redis
    ports:
      - "127.0.0.1:6379:6379"   # 仅绑定本机，防止 Redis 对公网暴露
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

volumes:
  redis_data:
```

---

## 十二、常见问题排查

| 现象 | 排查步骤 |
|------|---------|
| `docker compose build` 失败 | 检查网络是否能访问 PyPI，尝试配置镜像加速 |
| `curl /api/v1/health` 返回 `indexing` | 服务正在启动，等待约1分钟后重试 |
| `STARTUP FAILED — Redis unavailable` | `docker compose ps` 检查 redis 容器是否健康 |
| 页面能访问但回复很慢 | CPU 推理正常现象，首次问题冷启动需 10～30 秒 |
| `FileNotFoundError` 指向模型路径 | 检查 volumes 挂载路径是否与 `/opt/models/` 下目录名一致 |
| 端口 8181 无法访问 | 检查腾讯云安全组入站规则是否已放行 8181 端口 |

---

## 十三、不用 Docker：直接用 Conda 环境部署（推荐与本地保持一致）

如果你希望在服务器上使用和本地完全相同的 Conda 环境来运行项目（而不用 Docker），按以下步骤操作。

### 13.1 安装 Miniconda

`bash
# 下载 Miniconda 安装脚本（Linux x86_64）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 静默安装到 /opt/miniconda3
bash miniconda.sh -b -p /opt/miniconda3

# 添加到 PATH（写入 ~/.bashrc）
echo 'export PATH="/opt/miniconda3/bin:"' >> ~/.bashrc
source ~/.bashrc

# 验证安装
conda --version
`

### 13.2 创建与本地相同的 Conda 环境

`bash
# 创建名为 enterprise_knowledge_base 的 Python 3.11 环境（与本地一致）
conda create -n enterprise_knowledge_base python=3.11 -y

# 激活环境
conda activate enterprise_knowledge_base
`

### 13.3 安装依赖

> **注意**：服务器如果没有 GPU，需要先安装 CPU 版 PyTorch，再安装其余依赖。

**无 GPU（CPU 推理）：**
`bash
# 先安装 CPU 版 PyTorch（避免拉取巨大的 CUDA 版本）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# 再安装其余依赖
pip install --no-cache-dir -r requirements.txt
`

**有 GPU（CUDA 12.x）：**
`bash
# 先安装 CUDA 版 PyTorch
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 再安装其余依赖
pip install --no-cache-dir -r requirements.txt

# 验证 GPU 可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
`

> **网络慢？** 可加 PyPI 镜像加速：
> `bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> `

### 13.4 配置环境变量

`bash
cd /opt/enterprise_knowledge_base

# 从模板复制 .env
cp .env.example .env
nano .env
`

填写内容参考[第六章](#六配置环境变量)，模型路径改为服务器上的实际路径，例如：

`dotenv
EMBED_MODEL_NAME=/opt/models/bge-small-zh-v1.5
RERANK_MODEL_NAME=/opt/models/bge-reranker-base
DEVICE=cpu
REDIS_URL=redis://localhost:6379/0
`

### 13.5 安装并启动 Redis

Conda 环境部署时需要单独运行 Redis（不依赖 Docker Compose）：

`bash
# 方式A：用 Docker 单独跑 Redis（推荐）
docker run -d --name ekb_redis \
  -p 127.0.0.1:6379:6379 \
  --restart unless-stopped \
  redis/redis-stack-server:latest

# 方式B：直接安装 Redis
sudo apt-get install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server
`

### 13.6 启动应用

`bash
# 确保在项目目录且已激活环境
cd /opt/enterprise_knowledge_base
conda activate enterprise_knowledge_base

# 前台运行（测试用）
uvicorn main:app --host 0.0.0.0 --port 8181

# 后台运行（生产用，日志写入 app.log）
nohup uvicorn main:app --host 0.0.0.0 --port 8181 > app.log 2>&1 &

# 查看日志
tail -f app.log
`

### 13.7 设置开机自启（systemd）

`bash
sudo tee /etc/systemd/system/ekb.service > /dev/null <<EOF
[Unit]
Description=Enterprise Knowledge Base App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/enterprise_knowledge_base
Environment="PATH=/opt/miniconda3/envs/enterprise_knowledge_base/bin"
ExecStart=/opt/miniconda3/envs/enterprise_knowledge_base/bin/uvicorn main:app --host 0.0.0.0 --port 8181
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ekb
sudo systemctl start ekb

# 查看服务状态
sudo systemctl status ekb
`

### 13.8 健康检查

`bash
curl http://localhost:8181/api/v1/health
# 期望：{"status":"ready","index_loaded":true,"doc_chunk_count":XX}
`

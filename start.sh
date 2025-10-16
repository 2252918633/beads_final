#!/bin/bash
# 拼豆图纸生成器 - 启动脚本
set -e
export http_proxy=http://10.0.1.151:7890
export https_proxy=http://10.0.1.151:7890

echo "======================================"
echo "  拼豆图纸生成器 - 启动脚本  "
echo "  自动检测并使用 docker compose 或 docker-compose  "
echo "======================================"
echo ""

COMPOSE_FILE="docker-compose.yml"
PORT="80"
ENV_NAME="生产环境"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装（兼容新旧版本）
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ 错误: 未检测到 Docker Compose"
    echo "请安装: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker Compose 已安装: 使用命令 '$COMPOSE_CMD'"

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p backend/uploads backend/outputs backend/temp_processing logs

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  未找到 .env 文件，使用默认配置"
    echo "   如需自定义配置，请复制 .env.example 为 .env 并修改"
fi

# 构建并启动容器
echo ""
echo "🔨 开始构建 Docker 镜像..."
$COMPOSE_CMD -f "$COMPOSE_FILE" build

echo ""
echo "🚀 启动服务..."
$COMPOSE_CMD -f "$COMPOSE_FILE" up -d

echo ""
echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
if $COMPOSE_CMD -f "$COMPOSE_FILE" ps | grep -q "Up"; then
    echo ""
    echo "✅ 服务启动成功！"
    echo ""
    echo "======================================"
    echo "  访问地址："
    if [ "$PORT" == "80" ]; then
        echo "  http://localhost"
        echo "  (如需 HTTPS，请配置 SSL 证书)"
    else
        echo "  http://localhost:$PORT"
    fi
    echo "======================================"
    echo ""
    echo "📋 常用命令："
    echo "  查看日志: $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
    echo "  停止服务: $COMPOSE_CMD -f $COMPOSE_FILE down"
    echo "  重启服务: $COMPOSE_CMD -f $COMPOSE_FILE restart"
    echo ""
else
    echo ""
    echo "❌ 服务启动失败，请查看日志："
    echo "   $COMPOSE_CMD -f $COMPOSE_FILE logs"
    exit 1
fi

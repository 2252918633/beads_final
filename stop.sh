#!/bin/bash
# 拼豆图纸生成器 - 停止脚本

set -e

echo "======================================"
echo "  拼豆图纸生成器 - 停止服务  "
echo "======================================"
echo ""

# 检查是否保留数据
if [ "$1" == "--clean" ]; then
    echo "⚠️  警告: 将删除所有数据（包括上传的图片和生成的图纸）"
    read -p "确认继续? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  停止服务并删除数据..."
        docker-compose down -v
        rm -rf backend/uploads/* backend/outputs/* backend/temp_processing/* logs/*
        echo "✅ 服务已停止，数据已清理"
    else
        echo "❌ 操作已取消"
        exit 0
    fi
else
    echo "🛑 停止服务（保留数据）..."
    docker-compose down
    echo "✅ 服务已停止"
    echo ""
    echo "💡 提示："
    echo "  重新启动: ./start.sh 或 docker-compose up -d"
    echo "  删除数据: ./stop.sh --clean"
fi

echo ""


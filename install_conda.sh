#!/bin/bash

echo "番茄检测器 - Conda 环境安装"

# 检查是否已安装 Conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 Conda。请先安装 Anaconda 或 Miniconda。"
    echo "下载地址: https://www.anaconda.com/download/ 或 https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "开始创建 Conda 环境 (tomato-detector)..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "错误: 环境创建失败。请检查 environment.yml 文件和网络连接。"
    exit 1
fi

echo ""
echo "环境创建成功！"
echo "请使用以下命令激活环境:"
echo "    source activate tomato-detector"
echo ""
echo "然后运行程序:"
echo "    python main.py"
echo ""

# 为脚本添加执行权限
chmod +x install_conda.sh

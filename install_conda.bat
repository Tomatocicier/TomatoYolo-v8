@echo off
echo 番茄检测器 - Conda 环境安装

REM 检查是否已安装 Conda
call conda --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未检测到 Conda。请先安装 Anaconda 或 Miniconda。
    echo 下载地址: https://www.anaconda.com/download/ 或 https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo 开始创建 Conda 环境 (tomato-detector)...
call conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo 错误: 环境创建失败。请检查 environment.yml 文件和网络连接。
    pause
    exit /b 1
)

echo.
echo 环境创建成功！
echo 请使用以下命令激活环境:
echo     conda activate tomato-detector
echo.
echo 然后运行程序:
echo     python main.py
echo.

pause

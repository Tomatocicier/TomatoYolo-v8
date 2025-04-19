import sys
import logging
import torch
from PyQt5.QtWidgets import QApplication
from tomato_detector import TomatoDetectorWindow

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tomato_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TomatoDetector.Main")

if __name__ == "__main__":
    logger.info("番茄检测应用程序启动")
    
    # 记录PyTorch版本和CUDA信息
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        logger.info(f"当前CUDA设备: {torch.cuda.current_device()}")
        logger.info(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    try:
        app = QApplication(sys.argv)
        window = TomatoDetectorWindow()
        window.show()
        logger.info("应用程序主窗口显示")
        exit_code = app.exec_()
        logger.info(f"应用程序结束，退出代码: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"应用程序崩溃: {str(e)}", exc_info=True)
        raise

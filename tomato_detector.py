import os
import sys
import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QMessageBox, 
                            QProgressBar, QStatusBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tomato_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TomatoDetector")

class ModelThread(QThread):
    """用于在后台线程中运行模型推理的线程类"""
    result_ready = pyqtSignal(object, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path, image_path, device='auto'):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.device = device  # 'auto', 'cpu', 或 'cuda:0'等GPU设备
        self.logger = logging.getLogger("TomatoDetector.ModelThread")
        
    def run(self):
        try:
            self.logger.info(f"开始加载模型: {self.model_path}")
            # 使用ultralytics库加载YOLOv8模型
            from ultralytics import YOLO
            
            self.logger.info("导入YOLO模型成功")
            # 强制使用CPU设备，避免CUDA错误
            model = YOLO(self.model_path)
            self.logger.info(f"模型加载成功: {type(model)}")
            
            # 检查CUDA可用性并记录设备信息
            cuda_available = torch.cuda.is_available()
            self.logger.info(f"CUDA是否可用: {cuda_available}")
            if cuda_available:
                self.logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"PyTorch CUDA版本: {torch.version.cuda}")
            
            self.logger.info(f"开始处理图像: {self.image_path}")
            
            # 根据用户选择使用指定的设备进行推理
            if self.device == 'auto':
                self.logger.info("使用自动选择的设备进行推理")
                results = model(self.image_path)
            else:
                self.logger.info(f"使用指定设备 '{self.device}' 进行推理")
                results = model(self.image_path, device=self.device)
            self.logger.info(f"图像处理完成，获取结果: {type(results)} 数量: {len(results)}")
            
            # 处理结果
            found_tomato = False
            confidence = 0.0
            
            for i, result in enumerate(results):
                self.logger.debug(f"处理结果 {i+1}/{len(results)}")
                boxes = result.boxes
                self.logger.debug(f"找到 {len(boxes)} 个检测框")
                
                for j, box in enumerate(boxes):
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    self.logger.debug(f"检测框 {j+1}: 类别ID={cls}, 置信度={conf:.4f}")
                    
                    # 记录模型类别名映射
                    if hasattr(model, 'names'):
                        self.logger.debug(f"模型类别映射: {model.names}")
                    
                    # 假设番茄的类别标签为0或者"tomato"
                    # 根据您的模型可能需要调整这部分
                    is_tomato = False
                    if cls == 0:
                        is_tomato = True
                        self.logger.debug("类别ID为0，识别为番茄")
                    elif hasattr(model, 'names') and model.names[cls].lower() == 'tomato':
                        is_tomato = True
                        self.logger.debug(f"类别名称为'{model.names[cls]}'，识别为番茄")
                    
                    if is_tomato:
                        found_tomato = True
                        if conf > confidence:
                            confidence = conf
                            self.logger.debug(f"更新最高置信度: {confidence:.4f}")
            
            # 处理结果图像
            self.logger.info("开始生成结果图像")
            result_img = results[0].plot()
            self.logger.info(f"结果图像生成成功: 形状={result_img.shape if isinstance(result_img, np.ndarray) else '未知'}")
            
            message = ""
            if found_tomato:
                message = f"检测到番茄! 置信度: {confidence:.2f}"
                self.logger.info(f"检测结果: {message}")
            else:
                message = "未检测到番茄"
                self.logger.info("检测结果: 未找到番茄")
                
            self.result_ready.emit(result_img, message)
            
        except ImportError as ie:
            error_msg = f"缺少必要的库: {str(ie)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
        except FileNotFoundError as fnf:
            error_msg = f"找不到文件: {str(fnf)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
        except torch.cuda.OutOfMemoryError as oom:
            error_msg = f"GPU内存不足: {str(oom)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
        except NotImplementedError as nie:
            if "torchvision::nms" in str(nie) and "CUDA" in str(nie):
                error_msg = "CUDA版本与PyTorch不兼容，请尝试使用CPU模式"
                self.logger.error(f"{error_msg}: {str(nie)}", exc_info=True)
                
                # 尝试使用CPU重新运行
                try:
                    self.logger.info("尝试使用CPU模式重新运行模型...")
                    from ultralytics import YOLO
                    model = YOLO(self.model_path)
                    # 无论用户选择什么，在这种错误情况下都强制使用CPU
                    results = model(self.image_path, device='cpu')
                    
                    # 处理结果
                    found_tomato = False
                    confidence = 0.0
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls.item())
                            conf = box.conf.item()
                            
                            if cls == 0 or (hasattr(model, 'names') and model.names[cls].lower() == 'tomato'):
                                found_tomato = True
                                confidence = max(confidence, conf)
                    
                    result_img = results[0].plot()
                    
                    message = ""
                    if found_tomato:
                        message = f"检测到番茄! 置信度: {confidence:.2f}"
                        self.logger.info(f"检测结果: {message}")
                    else:
                        message = "未检测到番茄"
                        self.logger.info("检测结果: 未找到番茄")
                        
                    self.result_ready.emit(result_img, message)
                    return
                except Exception as e2:
                    error_msg = f"CPU模式重试失败: {str(e2)}"
                    self.logger.error(error_msg, exc_info=True)
                    self.error_occurred.emit(error_msg)
            else:
                error_msg = f"操作未实现: {str(nie)}"
                self.logger.error(error_msg, exc_info=True)
                self.error_occurred.emit(error_msg)
        except Exception as e:
            error_msg = f"处理过程中出错: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)


class TomatoDetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.image_path = ""
        self.logger = logging.getLogger("TomatoDetector.Window")
        self.logger.info("初始化番茄检测器窗口")
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("番茄检测器")
        self.setGeometry(100, 100, 800, 600)
        self.logger.debug("UI初始化")
        
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 顶部按钮区域
        button_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("加载模型(.pt)")
        self.load_model_btn.clicked.connect(self.load_model)
        
        self.load_image_btn = QPushButton("加载图片")
        self.load_image_btn.clicked.connect(self.load_image)
        
        self.detect_btn = QPushButton("检测")
        self.detect_btn.clicked.connect(self.detect_tomato)
        self.detect_btn.setEnabled(False)
        
        button_layout.addWidget(self.load_model_btn)
        button_layout.addWidget(self.load_image_btn)
        button_layout.addWidget(self.detect_btn)
        
        # 设备选择区域
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QGroupBox
        device_group_box = QGroupBox("计算设备")
        device_layout = QHBoxLayout()
        
        self.device_group = QButtonGroup(self)
        
        self.auto_device_radio = QRadioButton("自动选择")
        self.cpu_device_radio = QRadioButton("仅CPU")
        self.gpu_device_radio = QRadioButton("GPU (CUDA)")
        
        # 默认选择自动
        self.auto_device_radio.setChecked(True)
        
        # 检查CUDA是否可用，如果不可用则禁用GPU选项
        cuda_available = torch.cuda.is_available()
        self.gpu_device_radio.setEnabled(cuda_available)
        if not cuda_available:
            self.gpu_device_radio.setToolTip("此设备上CUDA不可用")
        
        # 添加到按钮组
        self.device_group.addButton(self.auto_device_radio, 0)
        self.device_group.addButton(self.cpu_device_radio, 1)
        self.device_group.addButton(self.gpu_device_radio, 2)
        
        # 连接信号
        self.device_group.buttonClicked.connect(self.device_changed)
        
        # 添加到布局
        device_layout.addWidget(self.auto_device_radio)
        device_layout.addWidget(self.cpu_device_radio)
        device_layout.addWidget(self.gpu_device_radio)
        device_group_box.setLayout(device_layout)
        
        # 展示区域
        self.image_label = QLabel("在这里显示图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid #cccccc;")
        
        # 状态区域
        self.model_status = QLabel("模型状态: 未加载")
        self.image_status = QLabel("图片状态: 未加载")
        self.result_status = QLabel("检测结果: -")
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 添加所有组件到主布局
        main_layout.addLayout(button_layout)
        main_layout.addWidget(device_group_box)  # 添加设备选择框
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.model_status)
        main_layout.addWidget(self.image_status)
        main_layout.addWidget(self.result_status)
        main_layout.addWidget(self.progress_bar)
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        self.setCentralWidget(main_widget)
        self.logger.debug("UI初始化完成")
    
    def load_model(self):
        self.logger.info("开始加载模型")
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(self, "选择YOLOv8模型文件", "", "PyTorch 模型 (*.pt)")
        
        if model_path:
            self.model_path = model_path
            self.logger.info(f"已选择模型: {model_path}")
            self.model_status.setText(f"模型状态: 已加载 ({os.path.basename(model_path)})")
            self.statusBar.showMessage(f"模型已加载: {os.path.basename(model_path)}")
            self.check_detect_button()
        else:
            self.logger.info("用户取消了模型选择")
    
    def load_image(self):
        self.logger.info("开始加载图片")
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)")
        
        if image_path:
            self.image_path = image_path
            self.logger.info(f"已选择图片: {image_path}")
            self.image_status.setText(f"图片状态: 已加载 ({os.path.basename(image_path)})")
            self.statusBar.showMessage(f"图片已加载: {os.path.basename(image_path)}")
            
            # 显示图片
            self.display_image(image_path)
            self.check_detect_button()
        else:
            self.logger.info("用户取消了图片选择")
    
    def display_image(self, image_path, processed=False):
        self.logger.debug(f"正在显示{'处理后的' if processed else '原始'}图片")
        pixmap = QPixmap(image_path) if not processed or isinstance(image_path, str) else None
        
        if (pixmap is None or pixmap.isNull()) and processed:
            # 如果是处理后的图像，可能是OpenCV格式，需要转换
            try:
                self.logger.debug("处理OpenCV格式的图像结果")
                if isinstance(image_path, np.ndarray):
                    height, width, channel = image_path.shape
                    self.logger.debug(f"图像尺寸: {width}x{height}, 通道数: {channel}")
                    bytes_per_line = 3 * width
                    # OpenCV默认是BGR，需要转换为RGB
                    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
                    q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.logger.debug("OpenCV图像转换为QPixmap成功")
            except Exception as e:
                error_msg = f"无法加载处理后的图像: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                QMessageBox.warning(self, "图像加载错误", error_msg)
                return
        
        if pixmap and not pixmap.isNull():
            # 缩放图片适应label
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.logger.debug("图像显示成功")
        else:
            error_msg = f"无法加载图像: {image_path}"
            self.logger.error(error_msg)
            QMessageBox.warning(self, "图像加载错误", error_msg)
    
    def check_detect_button(self):
        # 只有当模型和图片都已加载时，才启用检测按钮
        if self.model_path and self.image_path:
            self.detect_btn.setEnabled(True)
            self.logger.debug("启用检测按钮")
        else:
            self.detect_btn.setEnabled(False)
            self.logger.debug("禁用检测按钮 - 缺少模型或图片")
            
    def device_changed(self, button):
        """处理设备选择变更"""
        if button == self.auto_device_radio:
            self.device = "auto"
            self.logger.info("设备设置为: 自动选择")
        elif button == self.cpu_device_radio:
            self.device = "cpu"
            self.logger.info("设备设置为: 仅CPU")
        elif button == self.gpu_device_radio:
            self.device = "cuda:0"  # 使用第一个CUDA设备
            self.logger.info("设备设置为: GPU (CUDA)")
        
        # 更新状态栏
        self.statusBar.showMessage(f"计算设备: {self.device}")
    
    def detect_tomato(self):
        if not self.model_path or not self.image_path:
            self.logger.warning("尝试在模型或图片未加载时进行检测")
            QMessageBox.warning(self, "警告", "请先加载模型和图片")
            return
        
        self.logger.info(f"开始番茄检测 - 模型: {self.model_path}, 图片: {self.image_path}")
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.statusBar.showMessage("正在处理图像...")
        self.detect_btn.setEnabled(False)
        
        # 创建并启动检测线程，传递用户选择的设备类型
        self.detection_thread = ModelThread(self.model_path, self.image_path, self.device)
        self.detection_thread.result_ready.connect(self.process_results)
        self.detection_thread.error_occurred.connect(self.handle_error)
        self.detection_thread.finished.connect(self.detection_completed)
        self.detection_thread.start()
        
        # 更新状态栏显示使用的设备
        self.statusBar.showMessage(f"正在使用 {self.device} 处理图像...")
        
        self.progress_bar.setValue(50)
        self.logger.debug("检测线程已启动，进度条更新到50%")
    
    def process_results(self, result_image, message):
        self.logger.info(f"收到检测结果: {message}")
        
        # 显示处理后的图像
        self.display_image(result_image, processed=True)
        
        # 更新结果状态
        self.result_status.setText(f"检测结果: {message}")
        self.statusBar.showMessage("检测完成")
        self.progress_bar.setValue(100)
        self.logger.info("检测流程完成")
    
    def handle_error(self, error_message):
        self.logger.error(f"检测过程中发生错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
        self.statusBar.showMessage("检测失败")
        self.progress_bar.setValue(0)
    
    def detection_completed(self):
        self.logger.debug("检测线程已完成")
        self.detect_btn.setEnabled(True)
        # 延迟隐藏进度条
        self.progress_bar.setValue(100)
        # 可以添加一个定时器延迟隐藏进度条

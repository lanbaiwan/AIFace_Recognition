# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from ui_form import Ui_AIface
from pred import init_pred
from pred import get_pred

class MainWindow(QMainWindow, Ui_AIface):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.fname = ''  # 定义为实例变量
        self.transform,self.model=init_pred()
        # 连接按钮的点击信号到相应的槽函数
        self.selectButton.clicked.connect(self.open_image_file)
        self.confirmButton.clicked.connect(self.confirm_action)

    def open_image_file(self):
        # 打开文件对话框，让用户选择图片
        self.fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image files (*.png *.jpg *.jpeg *.bmp)")
        if self.fname:
            pixmap = QPixmap(self.fname)
            # 调整图片大小以适应QLabel的尺寸
            scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # 显示调整大小后的图片
            self.imageLabel.setPixmap(scaled_pixmap)

    def confirm_action(self):

        self.validationLine.setText(str(get_pred(self.fname,self.transform,self.model)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())

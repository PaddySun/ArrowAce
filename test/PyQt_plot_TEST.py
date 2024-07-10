import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.sc)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.setWindowTitle("Dynamic Matplotlib in PyQt5")
        
        # 添加定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # 每1000毫秒（1秒）更新一次
        
        self.counter = 0  # 用于计数，模拟动态数据
        
        self.show()

    def update_plot(self):
        # 更新数据
        self.counter += 1
        x_data = range(self.counter)
        y_data = [i**2 for i in x_data]  # 以平方作为示例数据
        
        # 清除旧的图表，准备绘制新的数据
        self.sc.axes.clear()
        
        # 重新绘制新的数据
        self.sc.axes.plot(x_data, y_data)
        self.sc.draw()  # 通知canvas重绘

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
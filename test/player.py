from PyQt5.QtMultimediaWidgets import QVideoWidget  # 添加这一行来导入QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from ui_player import Ui_MainWindow  # 导入转换后的UI类
from PyQt5.QtCore import QRect,QUrl  # 添加这一行来导入QRect

class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 将self.videoWidget替换为QVideoWidget实例
        self.videoWidget = QVideoWidget(self.centralwidget)
        self.videoWidget.setGeometry(QRect(20, 10, 640, 360))  # 现在可以正确使用QRect了

        # 视频播放设置
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        video_url = QUrl.fromLocalFile('C:/Users/paddy/Desktop/ArrowAce/data/output_20240702152045.mp4')  # 将本地文件路径转换为QUrl
        self.mediaPlayer.setMedia(QMediaContent(video_url))  # 使用转换后的QUrl # 替换为视频文件路径
        self.mediaPlayer.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
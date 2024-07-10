# ArrowAce

本项目是用视觉检测技术，用以指导业余射箭者的软件。

## **安装指南**

请确保安装了requirements.txt所述的库

## **使用说明**

**MediapipeDetection.py**

编译运行将自动调取您默认摄像头画面，在画面中未检测到人体则不会展示画面，可在关闭plot图标窗口后按Esc键结束录制。您录制的画面与人体关节节点数据将保存在./data目录下。

**DataAnalysis.py**

编译运行将读取./data目录下最新的pose_analysis_data_XXXX.csv与video_output_XXXX.mp4的文件对，并将绘制默认15号点位（左手腕）的x与y坐标图。可通过更改play_video_with_landmark_plot()函数中的landmark_index值来更改显示的点位。

![mp点位](.\docs\mp点位.png)



## 开发日志

**此前** 依托matplotlib的人体关节节点检测，完成实时图像采集，射箭支数检测影像与数据的格式化保存（MediapipeDetection.py），与事后对数据进行可视化分析（DataAnalysis.py）

实际测试下 射箭支数检测 可达90%左右，但数据可视化还不够直观。但还未想到好的方案。

**24-7-7** 确定使用freeze作为Web应用框架继续开发
**24-7-10** 整理项目文档.\ArrowAce\ArrowAce下将会把现有功能在webui上实现。.\ArrowAce\test保存了此前测试和实验的所有内容。






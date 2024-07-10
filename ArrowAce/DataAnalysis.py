import pandas as pd

import os
import re
from datetime import datetime

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import ast
import matplotlib.animation as animation

import threading

# pose_analysis_data_\d{14}.csv 数据格式
# ["FrameStamp", "TimeStamp", "State", "landmarks", "Confidence"]
   
def load_and_process_data():
	# 寻找符合要求的文件加载
	# 定义文件模式
	csv_pattern = re.compile(r'pose_analysis_data_(\d{8}\d{6})\.csv')
	mp4_pattern = re.compile(r'video_output_(\d{8}\d{6})\.mp4')

	# 存储文件名及其时间戳
	csv_files = []
	mp4_files = []

	# 遍历data文件夹
	for filename in os.listdir('data'):
		if csv_pattern.match(filename):
			# 提取时间戳并存储
			timestamp = csv_pattern.search(filename).group(1)
			csv_files.append((datetime.strptime(timestamp, '%Y%m%d%H%M%S'), filename))
		elif mp4_pattern.match(filename):
			# 提取时间戳并存储
			timestamp = mp4_pattern.search(filename).group(1)
			mp4_files.append((datetime.strptime(timestamp, '%Y%m%d%H%M%S'), filename))

	# 按时间戳降序排列
	csv_files.sort(reverse=True)
	mp4_files.sort(reverse=True)

	# 找到最新的匹配对
	latest_csv = csv_files[0][1] if csv_files else None
	latest_mp4 = mp4_files[0][1] if mp4_files else None

	# 确保两个文件的时间戳一致
	if latest_csv and latest_mp4:
		csv_timestamp = csv_pattern.search(latest_csv).group(1)
		mp4_timestamp = mp4_pattern.search(latest_mp4).group(1)
		if csv_timestamp == mp4_timestamp:
			print(f"最新匹配的文件对:")
			print(f"CSV: {latest_csv}")
			print(f"MP4: {latest_mp4}")
		else:
			print("没有找到时间戳一致的最新CSV和MP4文件对。")
	else:
		print("没有找到符合条件的文件。")

	return f"{latest_csv}", f"{latest_mp4}"


def play_video_with_landmark_plot(csv_file, video_path, landmark_index, window_title='Landmark Position Over Time'):
	# 初始化视频捕捉
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print("无法打开视频文件")
		return
	
	# 读取CSV数据
	df = pd.read_csv(csv_file)
	landmarks_col = df['landmarks'].apply(ast.literal_eval)
	
	# 准备绘图
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.set_title(window_title)
	ax.set_xlabel('Frame Number')
	ax.set_ylabel('Coordinate Value')
	x_data = []  # 存储帧号
	y_data_x = []  # 存储landmark的x坐标
	y_data_y = []  # 存储landmark的y坐标
	line_x, = ax.plot([], [], label=f'Landmark X-{landmark_index}')
	line_y, = ax.plot([], [], label=f'Landmark Y-{landmark_index}')
	ax.legend()
	
	def update(frame_number):
		ret, frame = cap.read()
		if not ret:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开始
			ret, frame = cap.read()

		cv2.imshow('Pose', frame)
		# 获取当前帧对应的地标数据
		current_landmarks = landmarks_col[frame_number]
		if len(current_landmarks) > landmark_index:
			x_data.append(frame_number)
			y_data_x.append(float(current_landmarks[landmark_index][0]))
			y_data_y.append(float(current_landmarks[landmark_index][1]))
			
			line_x.set_data(x_data, y_data_x)
			line_y.set_data(x_data, y_data_y)
			ax.relim()  
			ax.autoscale_view(True,True,True)
			return line_x, line_y,
		else:
			# 不添加新数据，但可能需要更新显示范围以确保最后一个点可见
			ax.set_xlim([0, frame_number + 1])  # 或使用适当的范围调整逻辑
			ax.set_ylim([min(y_data_y), max(y_data_y)])  # 同样地，调整y轴范围
	 

	ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50, blit=True)

	plt.show()

	cap.release()
	cv2.destroyAllWindows()

csv_file, mp4_file = load_and_process_data()
play_video_with_landmark_plot(os.path.join('data', csv_file), os.path.join('data', mp4_file), 16)

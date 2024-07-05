import pandas as pd

import os
import re
from datetime import datetime


def load_and_process_data():
	# 寻找符合要求的文件加载
	# 定义文件模式
	csv_pattern = re.compile(r'pose_analysis_data_(\d{8}\d{6})\.csv')
	mp4_pattern = re.compile(r'output_(\d{8}\d{6})\.mp4')

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

load_and_process_data()
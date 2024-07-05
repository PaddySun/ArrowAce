import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import time
import numpy as np
from google.protobuf.json_format import MessageToJson
import datetime
import pandas as pd

import pyttsx3 #语言播报库
import threading #线程分配

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# 绘制3D视图
colorclass = plt.cm.ScalarMappable(cmap='jet')
colors = colorclass.to_rgba(np.linspace(0, 1, int(33)))
colormap = (colors[:, 0:3])

# 姿态记录类，记录关键节位置点，判断状态
class LandmarkData:
	def __init__(self, timestamp, landmarks_result):
		self.time = timestamp # 记录此帧的时间
		#self.result = landmarks_result
		self.landmarks = np.array([[lmk.x, lmk.y] for lmk in landmarks_result.pose_world_landmarks.landmark]) # 各节点坐标与置信度
		self.confidences = np.array([[lmk.visibility] for lmk in landmarks_result.pose_world_landmarks.landmark])

		# 关键关节点
		self.L_wrist = self.dictionary_generation(16)  # 左手腕
		self.L_shoulder = self.dictionary_generation(12)  # 左肩
		self.R_wrist = self.dictionary_generation(15)
		self.R_shoulder = self.dictionary_generation(11)

	# 关键关节字典生成
	def dictionary_generation(self, n):
		return {"time": self.time, "landmarks" : self.landmarks[n], "confidences" : self.confidences[n][0]}

	# 两点间距离计算 
	def displacement_calculation(self, landmarks, previous_LandmarkData):
		current_coords = landmarks  # 选取x, y, z坐标
		previous_coords = previous_LandmarkData.landmarks  # 同样选取x, y, z坐标
		displacement = np.linalg.norm(current_coords - previous_coords, axis=1)
		return displacement

	# 计算所有点的速度绝对值的加权合
	def V_join_calculation(self, previous_LandmarkData):
		time_gap = self.time - previous_LandmarkData.time
		if time_gap == 0:
			return -1 #返回错误值

		dis = self.displacement_calculation(self.landmarks, previous_LandmarkData)
		dis = np.abs(dis)

		weighted_speeds = dis / time_gap * (self.confidences * previous_LandmarkData.confidences)

		V_join = np.sum(weighted_speeds)

		return V_join

	# 计算单点的加权速度绝对值
	def V_calculation(self, point1, point2):
		time_gap = point1["time"]-point2["time"]
		dis = np.linalg.norm(point1["landmarks"] - point2["landmarks"])
		dis = np.abs(dis)

		V = dis / time_gap * (point1["confidences"] * point2["confidences"])
		return V

	# 两点沿x/y/z坐标轴距离计算
	def height_dis_calculation(self, n, point1, point2):
		# x, y, z轴为0,1,2
		dis = point1["landmarks"][n] - point2["landmarks"][n]
		return dis

	# 状态判断
	def state_judgment(self, previous_LandmarkData):

		# 阈值
		# 静止
		stationary_V_join = [0,30] #全身运动阈值
		# 开弓
		L_wrist_v = 0.07 #右手手腕移动速度阈值越小越精确
		R_wrist_v = 0.03 #左手手腕移动速度阈值越小越精确
		# 瞄准
		L_wrist2shoulder_dis = 0.19 #右手手腕与右肩的距离 越小越精确
		Wrist_Diff = [-0.02,0.02] #左手腕与左肩的距离 越小越精确


		# 状态向量 依次为 静止 开工 瞄准
		self.state = -1
		# 静止
		V_join = self.V_join_calculation(previous_LandmarkData)
		if stationary_V_join[0] < V_join < stationary_V_join[1]:
			self.state = 0
		
		# 开弓
		L_v = self.V_calculation(self.L_wrist, previous_LandmarkData.L_wrist)
		R_v = self.V_calculation(self.R_wrist, previous_LandmarkData.R_wrist)
		if L_v > L_wrist_v and R_v > R_wrist_v:
			self.state = 1

		# 瞄准
		L_wrist2shoulder = np.linalg.norm(self.L_wrist["landmarks"] - self.L_shoulder["landmarks"])
		wrist_diff = self.height_dis_calculation(1, self.R_wrist , self.R_shoulder)
		if  L_wrist2shoulder < L_wrist2shoulder_dis and Wrist_Diff[0] < wrist_diff < Wrist_Diff[1]:
			self.state = 2
		# 错误检测
		if V_join == -1:
			self.state = -1

		state_measure = [V_join, L_v, R_v, L_wrist2shoulder, wrist_diff] # 将检测状态的量打包

		return self.state, state_measure

	# 判断是否完整完成了射箭动作
	def archery_judgment(self, action_time, current_time):
		maxtime_for_single_shot = 60
		mintime_for_single_shot = 6

		archery = False
		if self.state == 2 and current_time - action_time[2] < maxtime_for_single_shot:
			#速射不预瞄
			snapshot = all(action_time[i] < action_time[i + 1] for i in range(len(action_time) - 2)) and action_time[2] - action_time[0] < 60 
			#预瞄，开工后停止再到靠位
			pre_acquisition = action_time[1] < action_time[2] and action_time[0] < action_time[2] and action_time[2] - action_time[0] > 2 and action_time[2] - action_time[0] < 60
			if snapshot or pre_acquisition:
				# 
				if current_time-action_time[3] > mintime_for_single_shot :
					archery = True
		return archery

# 绘制图像
class DrwaImage:
	def __init__(self, undrwa_image, input_results ,input_start_time):
		# 在图像上绘制姿势注释
		self.image = undrwa_image
		self.results = input_results
		self.start_time = input_start_time

	# 绘制节点
	def Node2image(self, node_image):
		node_image.flags.writeable = True
		node_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
		mp_drawing.draw_landmarks(
			node_image,
			self.results.pose_landmarks,
			mp_pose.POSE_CONNECTIONS,
			landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())	
		return node_image

	# 绘制fps值
	def fps2image(self, fps_image):
		# 计算FPS
		end_time = time.time()
		fps = 1 / (end_time - self.start_time)
		fps = "%.2f fps" % fps
		# 实时显示帧数
		fps_image = cv2.flip(fps_image, 1)
		cv2.putText(fps_image, "FPS {0}".format(fps), (100, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),3)  
		return fps_image

	def draw(self, mode = 1):
		# mode 值
		# 其他 原图直出！ 
		# 1 仅绘制节点
		# 2 绘制节点与fps值
		# 3	
		if mode == 1: 
			drawed_image = self.Node2image(self.image)
		elif mode == 2: 
			node_image = self.Node2image(self.image)
			drawed_image = self.fps2image(node_image)
		elif mode == 3: 
			pass
		else: #原图直接出！
			drawed_image = self.image
		cv2.imshow('MediaPipe Pose', drawed_image)

# 绘制统计图
class DeawPlot:
	def __init__(self, mode ,colors = ['red', 'green', 'blue']):
		self.fig1, self.plot_plot1 = plt.subplots(figsize=(8, 4))
		self.plot_plot1.set_xlabel('Frame')
		self.plot_plot1.set_ylabel('Actions')
		self.plot_plot1.plot([], [])  # 初始化空图
		
		self.fig2, self.plot_plot2 = plt.subplots(figsize=(8, 4))
		self.plot_plot2.set_xlabel('Frame')
		self.plot_plot2.set_ylabel('Action Timing')
		self.plot_plot2.plot([], [])  # 初始化空图
		
		self.fig3d = plt.figure()
		self.ax_3d = self.fig3d.add_subplot(111, projection="3d")
		# self.ax_3d.set_xlim3d(-1, 1)
		# self.ax_3d.set_ylim3d(-1, 1)
		# self.ax_3d.set_zlim3d(-1, 1)
		
		self.colors = colors
		self.mode = mode


		# 显示但不阻塞
		plt.ion()
		if mode[0] == 1:
			self.fig1.show()
		else:
			plt.close(self.fig1)
		if mode[1] == 1:
			self.fig2.show()
		else:
			plt.close(self.fig2)
		if mode[2] == 1:
			self.fig3d.show()
		else:
			plt.close(self.fig3d)

	def preprocessing(self, plot_lis):
		if len(plot_lis) > 75: # 限制数据量
			plot_lis.pop(0)
			for i in range(len(plot_lis)):
				plot_lis[i] = (plot_lis[i][0] - 1, plot_lis[i][1])
		return plot_lis
	
	# 绘制1个数据的折线图
	def plot1(self, plot_lis):
		self.mode[0] = 1
		self.plot_plot1.clear()  # 清除旧图
		x, y = zip(*plot_lis)  # 解压列表为x轴和y轴数据
		self.plot_plot1.plot(x, y, color=self.colors[0])
		self.fig1.canvas.draw_idle()  # 更新显示
		plt.pause(0.001)


	# 绘制3个数据的折线图
	def plot3(self, plot_lis):
		self.mode[1] = 1
		x, y = zip(*plot_lis)  # 解压列表为x轴和y轴数据
		ys = [[item[1][i] for item in plot_lis] for i in range(3)]  # 分别对应多个y值的情况
		self.plot_plot2.clear()  # 清除旧图
		for i, y in enumerate(ys):
			self.plot_plot2.plot(x, y, color=self.colors[i], label=f"Action {i+1}")
		#plot_plot.plot(x, y, color='red')
		self.fig2.canvas.draw_idle()  # 更新显示
		plt.pause(0.001)

	# 绘制3D骨骼图
	def draw3d(self, world_landmarks, connections=mp_pose.POSE_CONNECTIONS):
		self.ax_3d.clear()
		self.mode[2] = 1
		self.ax_3d.set_xlim3d([-1, 1])
		self.ax_3d.set_ylim3d([-1, 1])
		self.ax_3d.set_zlim3d([-1, 1])

		landmarks = []
		for index, landmark in enumerate(world_landmarks.landmark):
		    landmarks.append([landmark.x, landmark.z, landmark.y*(-1)])
		landmarks = np.array(landmarks)

		self.ax_3d.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.array(colormap), s=50)
		for _c in connections:
		    self.ax_3d.plot([landmarks[_c[0], 0], landmarks[_c[1], 0]],
						    [landmarks[_c[0], 1], landmarks[_c[1], 1]],
						    [landmarks[_c[0], 2], landmarks[_c[1], 2]], 'k')
		self.fig3d.canvas.draw_idle()  # 更新显示
		plt.pause(0.001)

# 语言播报模块
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 空列表用于存储每帧的landmarks数据
landmarks_data = []

# 前一帧信息
previous_LandmarkData = None

# 状态时间
action_time = [0,0,0,0]

# 帧数
N = 0 #总帧数

# 初始化绘图函数
deaw_plot = DeawPlot([0,1,0])
# 用于绘图的数据
plot_lis1 = []
plot_lis2 = []

# 关键数据记录
# 射箭支数
shot_N = 0
# 射出时间点
shot_time_list = []

# 获取视频流
# 端口号一般是0，除非你还有其他摄像头
cap = cv2.VideoCapture(0)
# 使用本地视频推理，复制其文件路径代替端口号即可
#cap = cv2.VideoCapture("data/全身 少部遮挡.mp4")

with mp_pose.Pose(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5, 
	model_complexity = 1) as pose:

	# fig3D = plt.figure()
	# ax = fig3D.add_subplot(111, projection="3d")
	
	program_start_time = time.time()

	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue
			#break
		
		# 处理帧用于定位
		start_time = time.time()
		image.flags.writeable = False # 为提高性能，可选择将图像标记为不可写入，以 通过引用传递。
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = pose.process(image)
		
		# 构成当前帧点位数据 landmarks
		if results.pose_world_landmarks:
			# 执行状态判断
			# 构建 姿态记录类
			current_LandmarkData = LandmarkData(start_time, results)
			if previous_LandmarkData is not None:
				N += 1

				current_state, current_state_measure= current_LandmarkData.state_judgment(previous_LandmarkData)

				correct_status = -1
				# 动作变化的时间节点记录
				if current_state != -1:
					action_time[current_state] = start_time

				# 根据时间进一步判断状态，是否有必要？
					if current_state == 0 and start_time - action_time[0] >= 1:
						corrected_state = 0
					if current_state == 1 and start_time - action_time[1] >= 1:
						corrected_state = 1
					if current_state == 2 and start_time - action_time[1] >= 1:
						corrected_state = 2
				if correct_status != -1:
					action_time[corrected_state] = start_time

				# 射击判断
				if current_LandmarkData.archery_judgment(action_time, start_time):
					current_state = 3
					action_time[3] = start_time

					shot_N += 1 #射出支数+1
					shot_time_list.append = start_time #记录射箭时间
					speak_text(shot_N) #语言播报支数统计

				# 绘制图像
				drwa_image = DrwaImage(image, results ,start_time)
				drwa_image.draw(2)

				# 绘制统计图
				# 构建数据
				plot_lis1.append((N, (int(program_start_time - action_time[0])%1000,int(program_start_time - action_time[1])%1000,int(program_start_time - action_time[2])%1000)))
				plot_lis2 = deaw_plot.preprocessing(plot_lis1)

				plot_lis2.append((N, current_state_measure[4])) 
				plot_lis2 = deaw_plot.preprocessing(plot_lis2) 

				# 执行绘图 绘制哪几个需要调整deaw_plot
				deaw_plot.plot3(plot_lis1)
				deaw_plot.plot1(plot_lis2)
				deaw_plot.draw3d(results.pose_world_landmarks)
			
			previous_LandmarkData = current_LandmarkData

		if cv2.waitKey(5) & 0xFF == 27:
			break

# ... 释放资源、关闭窗口的部分 ...
print(shot_N)
# 在循环结束后释放资源
cap.release()
cv2.destroyAllWindows()
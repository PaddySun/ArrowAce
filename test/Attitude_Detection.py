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

colorclass = plt.cm.ScalarMappable(cmap='jet')
colors = colorclass.to_rgba(np.linspace(0, 1, int(33)))
colormap = (colors[:, 0:3])



# 语言播报模块
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def draw3d(plt, ax, world_landmarks, connnection=mp_pose.POSE_CONNECTIONS):
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    landmarks = []
    for index, landmark in enumerate(world_landmarks.landmark):
        landmarks.append([landmark.x, landmark.z, landmark.y*(-1)])
    landmarks = np.array(landmarks)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.array(colormap), s=50)
    for _c in connnection:
        ax.plot([landmarks[_c[0], 0], landmarks[_c[1], 0]],
                [landmarks[_c[0], 1], landmarks[_c[1], 1]],
                [landmarks[_c[0], 2], landmarks[_c[1], 2]], 'k')

    plt.pause(0.001)

#端口号一般是0，除非你还有其他摄像头
cap = cv2.VideoCapture(0)
#使用本地视频推理，复制其文件路径代替端口号即可
#cap = cv2.VideoCapture("data/全身 少部遮挡.mp4")

# 初始化视频保存
# 获取视频的宽度、高度和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
# 获取时间，作为视频文件命名
current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')  # 格式化当前时间为字符串
output_filename = f'output_{current_time}.mp4'
# 初始化VideoWriter对象，这里使用'mp4v'作为编码器，您可能需要根据系统支持选择合适的编码器
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (width, height)) 
print(f"Video saved to {output_filename}")

# 初始化一个空列表用于存储每帧的landmarks数据
landmarks_data = []


# 上一帧的关键点位置，初始化为空
previous_landmarks = None
displacement = None

# 定义左手腕的索引（注意，索引从0开始）
LEFT_WRIST_INDEX = 15

# 定义动作链
action = 0
action_time = [0,0,0,0] # 站定，举弓，瞄准，“达到撒放”的时间节点 最近一次完成动作的时间

# 帧计数
n = 0 #临时
N = 0 #总帧数
arrowsCount = 0 #射箭支数

# 举弓达顶点
d = 0

# 实现数据图标实时显示
# 初始化一个列表用于存储height_diff数据
plot_lis = []

# 添加一个子图用于显示height_diff
fig_height_diff, plot_plot = plt.subplots(figsize=(8, 4))
plot_plot.set_xlabel('Frame')
plot_plot.set_ylabel('Height Difference')
(plot_plot.plot([], []))  # 初始化空图

# 初始化一个定时器，用于实时更新height_diff的显示
timer = fig_height_diff.canvas.new_timer(interval=100)  # 每100毫秒更新一次
timer.add_callback(lambda event: plot_plot.relim(), fig_height_diff.canvas)
timer.add_callback(lambda event: plot_plot.autoscale_view(True,True,True), fig_height_diff.canvas)
timer.start()


# 分配语言播报模块到新线程
# 创建一个新的线程来运行speak_text函数
t = threading.Thread(target=speak_text, args=("start"))
t.start()# 开始新线程


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5, 
    model_complexity = 1) as pose:
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection="3d")

  while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
        #break
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    start = time.time()
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    

    # 人物姿态检测：静止
    # 如果有检测结果
    if results.pose_world_landmarks:
        # 当前帧的关键点位置
        current_landmarks = np.array([[lmk.x, lmk.y]
                                  for lmk in results.pose_world_landmarks.landmark])

        L_wrist_current = current_landmarks[16]  # 左手腕
        L_shoulder_current = current_landmarks[12]  # 左肩
        R_wrist_current = current_landmarks[15]
        R_shoulder_current = current_landmarks[11]

        # 确保上一帧的关键点位置已经被初始化
        if previous_landmarks is not None:
            # 计算关键点位移
            displacement = np.linalg.norm(current_landmarks - previous_landmarks, axis=1)
            #print(displacement[LEFT_WRIST_INDEX])
            
            # 静止检测
            # 检查是否所有关键点位移都小于阈值
            sum_below_threshold = np.sum(displacement)
            if np.all(displacement < 0.03):  # 阈值设为0.01，根据实际情况调整
                n += 1 #帧计数
            else:
                n = 0
                action = 0
            #print(n)
            if n > 2 :
                # 执行语音播报
                #speak_text("STOP")
                #print("STOP")
                action = 1
                if start-action_time[0] > 4 :
                    action_time[0] = start

            # # # 更新上一帧的关键点位置

            # # 举弓检测
            # # 检查手腕移动超过阈值
            if displacement[LEFT_WRIST_INDEX] > 0.07 and displacement[LEFT_WRIST_INDEX] < 0.2:  # 阈值根据实际情况调整
                # 执行语音播报
                #speak_text("RISE!")
                #print("RISE!")
                action = 2
                if start-action_time[1] > 2:
                    action_time[1] = start
 

            # 瞄准检测
            # 检测左手是否接近左肩高度 并且右手距离右肩够近(处于靠头一侧)
            height_diff = abs(R_wrist_current[1] - R_shoulder_current[1])
            dis = np.linalg.norm(L_wrist_current - L_shoulder_current)
            #print(L_wrist_current[0] - L_shoulder_current[0],L_wrist_current[1] - L_shoulder_current[1])
            if height_diff < 0.1 and dis < 0.13 and L_wrist_current[0] - L_shoulder_current[0] >= 0 and L_wrist_current[1] - L_shoulder_current[1] <= 0:
                # 执行语音播报
                # speak_text("Aim!")
                # print("Aim!")
                action = 3
                if start-action_time[2] > 2 :
                    action_time[2] = start

            # 完整动作检验
            print(int(action_time[0])%1000,int(action_time[1])%1000,int(action_time[2])%1000)
            snapshot = all(action_time[i] < action_time[i + 1] for i in range(len(action_time) - 2)) and action_time[2] - action_time[0] < 60 #速射不预瞄
            pre_acquisition = action_time[1] < action_time[2] and action_time[0] < action_time[2] and action_time[2] - action_time[0] > 2 and action_time[2] - action_time[0] < 60#预瞄，开工后停止再到靠位
            if snapshot or pre_acquisition:  
            #if action_time[1] < action_time[2] and action_time[2] - action_time[0] < 60:  
                    #校验动作依次完成
                    #校验动作在最大时间内完成
                if start-action_time[3] > 6 :
                    action_time[0] = start + 1
                    action_time[3] = start
                    arrowsCount += 1

                    speak_text(arrowsCount)

            plot_lis.append((N, (int(action_time[0])%1000,int(action_time[1])%1000,int(action_time[2])%1000)))  # 记录帧数和高度差
            if len(plot_lis) > 75: # 限制数据量
                plot_lis.pop(0)
                for i in range(len(plot_lis)):
                    plot_lis[i] = (plot_lis[i][0] - 1, plot_lis[i][1])

            colors = ['red', 'green', 'blue']  # 分别对应多组数据的颜色

            # 更新统计图
            x, y = zip(*plot_lis)  # 解压列表为x轴和y轴数据

            ys = [[item[1][i] for item in plot_lis] for i in range(3)]  # 分别对应多个y值的情况
            plot_plot.clear()  # 清除旧图
            for i, y in enumerate(ys):
                plot_plot.plot(x, y, color=colors[i], label=f"Action {i+1}")
            #plot_plot.plot(x, y, color='red')
            fig_height_diff.canvas.draw_idle()  # 更新显示
            plt.pause(0.001)
        #更新点位
        previous_landmarks = current_landmarks

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    end = time.time()
    fps = 1 / (end - start)
    fps = "%.2f fps" % fps
    N += 1
    #实时显示帧数
    image = cv2.flip(image, 1)
    cv2.putText(image, "FPS {0}".format(fps), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),3)   
    
    # 实时计算展示
    cv2.imshow('MediaPipe Pose', image)

    # 将处理后的帧写入输出文件
    out.write(image)

    if results.pose_world_landmarks:
        # 将当前帧的landmarks数据添加到列表
        landmarks_per_frame = []
        for landmark in results.pose_world_landmarks.landmark:
            landmarks_per_frame.extend([landmark.x, landmark.y, landmark.z])
        landmarks_data.append(landmarks_per_frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    # 绘制3D骨骼图
    # if results.pose_world_landmarks:
    #    draw3d(plt, ax, results.pose_world_landmarks)

    # ... 释放资源、关闭窗口的部分 ...

# # 循环结束后，将数据写入Excel
# # 将列表转换为DataFrame
# df = pd.DataFrame(landmarks_data, columns=[f"Landmark_{i}_X" for i in range(1, 34)] + 
#                   [f"Landmark_{i}_Y" for i in range(1, 34)] + 
#                   [f"Landmark_{i}_Z" for i in range(1, 34)])

# # 写入Excel文件
# df.to_excel(f"landmarks_{current_time}.xlsx", index=False)
# print(f"Landmarks data saved to landmarks_{current_time}.xlsx")

# 在循环结束后释放资源

plt.close(fig_height_diff) #释放matplotlib相关的资源
cap.release()
out.release()  # 释放VideoWriter
cv2.destroyAllWindows()

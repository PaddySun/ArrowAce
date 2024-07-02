import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import time
import numpy as np
from google.protobuf.json_format import MessageToJson
import datetime
import pandas as pd

import pyttsx3 #语言播报库

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

colorclass = plt.cm.ScalarMappable(cmap='jet')
colors = colorclass.to_rgba(np.linspace(0, 1, int(33)))
colormap = (colors[:, 0:3])

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


# 初始化语音引擎
engine = pyttsx3.init()

# 上一帧的关键点位置，初始化为空
previous_landmarks = None
displacement = None

# 定义左手腕的索引（注意，索引从0开始）
LEFT_WRIST_INDEX = 15

# 定义动作链
action_chain = []

# 帧计数
n = 0

# 举弓达顶点
d = 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5, 
    model_complexity = 1) as pose:
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")

  while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        #continue
        break
    
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
        current_landmarks = np.array([[lmk.x, lmk.y, lmk.z]
                                  for lmk in results.pose_world_landmarks.landmark])

        left_wrist_current = current_landmarks[15]  # 左手腕
        left_shoulder_current = current_landmarks[10]  # 左肩

        # 确保上一帧的关键点位置已经被初始化
        if previous_landmarks is not None:
            # 计算关键点位移
            displacement = np.linalg.norm(current_landmarks - previous_landmarks, axis=1)
            #print(displacement[LEFT_WRIST_INDEX])
            
            
            # 静止检测
            # 检查是否所有关键点位移都小于阈值
            if np.all(displacement < 0.01):  # 阈值设为0.01，根据实际情况调整
                n += 1 #帧计数
            else:
            	n = 0
        
            if n == 5:
                # 执行语音播报
                engine.say("Body is still")
                engine.runAndWait()
                n = 0
                action_chain.append("still")
            # 更新上一帧的关键点位置

            # 举弓检测
            # 检查手腕移动超过阈值
            left_wrist_current = current_landmarks[15]  # 左手腕
            left_shoulder_current = current_landmarks[10]  # 左肩
            if displacement[LEFT_WRIST_INDEX] > 0.1:  # 阈值根据实际情况调整
                # 执行语音播报
                engine.say("Bow mioving detected!")
                engine.runAndWait()

            # 瞄准检测
            # 检测左手是否接近左肩高度
            height_diff = abs(left_wrist_current[1] - left_shoulder_current[1])
            #print(n,height_diff)
            if height_diff < 0.1 and n > 0:
                # 执行语音播报
                engine.say("Aim!")
                engine.runAndWait()            	

    previous_landmarks = current_landmarks

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    end = time.time()
    fps = 1 / (end - start)
    fps = "%.2f fps" % fps
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
    #if results.pose_world_landmarks:
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
cap.release()
out.release()  # 释放VideoWriter
cv2.destroyAllWindows()

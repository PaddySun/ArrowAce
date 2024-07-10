from flask import Flask, render_template, Response, jsonify
import cv2

app = Flask(__name__)

# 初始化标志，用于控制摄像头的开启与关闭
camera_active = False

def gen_frames():
    global camera_active
    cap = cv2.VideoCapture(0)
    print(1)
    while True:
        if not camera_active:
            # 当摄像头未激活时，读取默认视频文件
            cap.open('imge.mp4')
        else:
            # 当摄像头激活时，读取摄像头画面
            cap.open(0)
        
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'status': 'success', 'active': camera_active})

if __name__ == '__main__':
    app.run(debug=True)
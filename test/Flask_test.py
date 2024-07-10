# app.py
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

@app.route('/call_function', methods=['POST'])
def my_function():
    # 这里是你的函数逻辑
    result = "按钮调用的函数执行完成！"
    return jsonify({"result": result})


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
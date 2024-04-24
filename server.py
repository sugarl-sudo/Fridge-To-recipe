from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from image_process import canny, ML
from datetime import datetime
import os
import string
import random

SAVE_DIR = "./images" # 画像保存ディレクトリ
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

app = Flask(__name__, static_url_path="")

# 長さnのランダムな文字列を生成
def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@app.route('/')
def index():
    images = os.listdir(SAVE_DIR)[::-1] # 画像PATHのリスト
    return render_template('index.html')

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

# 参考: https://qiita.com/yuuuu3/items/6e4206fdc8c83747544b
@app.route('/upload', methods=['POST'])
def upload():
    if request.files['image']:
        # 画像保存ディレクトリ内の画像を削除
        for filename in os.listdir(SAVE_DIR):
            file_path = os.path.join(SAVE_DIR, filename)
            os.unlink(file_path)

        # 入力画像をモデルが処理可能な形式に変換
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1) # Shape = (高さ, 幅, チャンネル数) dtype = uint8

        # 画像処理
        answer = ML(img)

        # 処理された画像を保存
        fixed_filename = "uploaded_image.png"
        save_path = os.path.join(SAVE_DIR, fixed_filename)
        cv2.imwrite(save_path, img)
        images = os.listdir(SAVE_DIR)[::-1]

        return render_template('index.html', images=images, answer=answer)
    
    return redirect('/')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8888)

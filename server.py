from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from image_process import ML
from chatgpt import chatgpt
import os

SAVE_DIR = "./uploaded_images" # 画像保存ディレクトリ
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

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

        # 入力画像から食材を検出
        food_names = ML(img)

        # 食材名のリストからプロンプトを生成
        prompt = f"{'、'.join(food_names)}を使った3つのレシピの料理名と材料と手順を挙げてください。材料の分量を必ず教えてください\
        出力のフォーマット：\
            ・料理名1\
                材料\
                手順\
            ・料理名2\
                材料\
                手順\
            ・料理名3\
                材料\
                手順\
            "
        
        # prompt = f"{'、'.join(food_names)}を使ったレシピの料理名を3つ挙げてください。このとき、番号は表示しないでください。\
        #     出力のフォーマット：\
        #         料理名\
        #         料理名\
        #         料理名\
        #         "

        # プロンプトからレシピを提案
        answer = chatgpt(prompt)
        answer = answer.replace(" ", "&nbsp;").replace('\n', '<br>')

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
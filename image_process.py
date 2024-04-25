import cv2
from datetime import datetime

# 仮の画像処理関数
def ML(image):
    answer = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") +'ChatGPTの回答'
    return answer

def canny(image):
    return cv2.Canny(image, 100, 200)
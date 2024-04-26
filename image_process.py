import cv2
from datetime import datetime
from classifier import run_classifier
from sam import run_sam
from ML import get_name


# 仮の画像処理関数
def ML(image):
    run_sam()
    segement_classes = run_classifier()
    food_names = [get_name(segement_class) for segement_class in segement_classes]
    # food_names = ['卵', '鶏肉','七味']
    return food_names

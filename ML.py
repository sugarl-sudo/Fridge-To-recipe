import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def job(filename, sam, ic, gpt3):
    print("==========================================================")
    print("おすすめレシピ")
    print("Configured in gpt-3.5-turbo, Segment Anything and resnet. ")
    print("==========================================================")
    # 画像の読み込み
    image, height, width = read_image(filename)
    print("read_image {} width:{} height:{}".format(filename, width, height))
    print("この写真を元に、おすすめのレシピを提案します。")
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    # マスク生成
    sam.generat_masks(image)

    data_list = []
    # 一定のサイズのものだけを抽出する
    max_pixels = 13000
    min_pixels = 2300

    for index in range(sam.length):
        mask, pixels = sam.get(index)

        # 一定範囲のサイズのみ対象にする
        if pixels < min_pixels or max_pixels < pixels:
            continue

        # 輪郭検出用の2値のテンポラリ画像
        mono_image = np.full(np.array([height, width, 1]), 255, dtype=np.uint8)
        # 個々の食品を切取るためのテンポラリ画像
        food_image = image.copy()

        area = Area(width, height)
        for y in range(height):
            for x in range(width):
                if mask[y][x]:
                    mono_image[y][x] = 0  # ２値画像は、マスク部分を黒にする
                    area.set(x, y)
                else:
                    food_image[y][x] = [255, 255, 255]  # 食品切取り画像は、マスク部分以外を白にする

        # 検出範囲
        x1, y1, x2, y2 = area.get()
        # 食品の輪郭を取得する
        contours, _ = cv2.findContours(mono_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 食品画像を切取る
        food_image = food_image[y1:y2, x1:x2]
        # 分類モデルで食品名を推論する
        class_name, probs = ic.inference(food_image)

        # nodeで始まる名前は、食品では無い
        if class_name.startswith("none") == False:
            data_list.append(Data(food_image, contours, class_name, probs))
            print("{} {:.2f}% [{},{},{},{}]".format(get_name(class_name), probs, x1, y1, x2, y2))

    # おすすめレシピ
    print("----------------------------------------------------------")
    food_list = []
    for data in data_list:
        food_list.append(get_name(data.class_name))

    gpt3.recipe_recommend(food_list)
    print("----------------------------------------------------------")

    # 元画像に輪郭を描画
    for data in data_list:
        cv2.drawContours(image, data.contours, -1, color=[0, 255, 255], thickness=3)
    plt.imshow(image)

    # 検出した食品画像を表示
    W = math.ceil(len(data_list) / 2)
    H = 2
    fig = plt.figure(figsize=(5, 5))

    for i, data in enumerate(data_list):
        ax1 = fig.add_subplot(H, W, i + 1)
        ax1.set_title("{} {:.2f}".format(get_name(data.class_name), data.probs), fontsize=10)
        plt.imshow(data.image)

    plt.axis("off")
    plt.show()

    # 画面クリア
    os.system("clear")

    # 名前の索引


def get_name(class_name):
    if class_name == "apple":
        return "りんご"
    elif class_name == "banana":
        return "バナナ"
    elif class_name == "Carrot":
        return "にんじん"
    elif class_name == "Cabbage":
        return "キャベツ"
    elif class_name == "Cauliflower":
        return "カリフラワー"
    elif class_name == "eggplant":
        return "ナス"
    elif class_name == "onion":
        return "玉ねぎ"
    elif class_name == "Potato":
        return "じゃがいも"
    elif class_name == "Tomato":
        return "トマト"
    elif class_name == "None":
        return ""
    else:
        raise ValueError("Unknown class name")


def read_image(file_name):
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    return image, height, width

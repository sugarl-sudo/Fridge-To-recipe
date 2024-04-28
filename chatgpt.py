import os
import requests
from dotenv import load_dotenv
import json

# 環境変数を読み込む
load_dotenv()

# APIキーを環境変数から取得
API_KEY = os.getenv("API_KEY")
print(f"API_KEY: {API_KEY}")

# プロンプトからレシピを提案する関数
def chatgpt(prompt):
    # エンドポイントへのPOSTリクエストのヘッダ
    header = {
        "Content-Type" : "application/json", # ボディのコンテンツタイプを指定
        "Authorization" : f"Bearer {API_KEY}",
    }

    # # エンドポイントへのPOSTリクエストのボディ
    # body = f'''
    # {{
    #     "model": "gpt-3.5-turbo",
    #     "temperature": 0,
    #     "messages": [
    #         {{"role": "user", "content": "{prompt}"}}
    #     ]
    # }}
    # '''
    # JSONデータを正しくエンコードするためにjson.dumpsを使用
    body = json.dumps({
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "top_p": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "seed": 42,
    })

    # エンドポイントからのPOSTレスポンス
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=header, data=body.encode('utf_8'))
    print(f"POSTレスポンスのJSONデータ: {response.text}")
    print(f"chat gpt response keys{response.json().keys()}")

    # デシリアライズ
    rj = response.json()

    answer = rj["choices"][0]["message"]["content"]
    answer = "・肉と玉ねぎの炒め物\n    材料：肉（200g）、玉ねぎ（1個）、しょうゆ（大さじ2）、砂糖（大さじ1）、酒（大さじ1）、サラダ油（大さじ1）\n    手順：\n    1. 肉を食べやすい大きさに切る。\n    2. 玉ねぎを薄切りにする。\n    3. フライパンにサラダ油を熱し、肉を炒める。\n    4. 肉に火が通ったら、玉ねぎを加えて炒める。\n    5. しょうゆ、砂糖、酒を加えて味を調える。\n    6. 火を止めて完成。\n\n・じゃがいもと玉ねぎのポトフ\n    材料：肉（300g）、じゃがいも（2個）、玉ねぎ（1個）、にんじん（1本）、キャベツ（1/4個）、トマト缶（1缶）、水（500ml）、塩、こしょう\n    手順：\n    1. 肉を食べやすい大きさに切る。\n    2. じゃがいも、玉ねぎ、にんじんを食べやすい大きさに切る。\n    3. フライパンで肉を焼き色がつくまで焼く。\n    4. 鍋に水とトマト缶を入れ、野菜と肉を加えて煮る。\n    5. 塩、こしょうで味を調えて完成。\n\n・ナスと玉ねぎの肉巻き\n    材料：肉（400g）、ナス（2本）、玉ねぎ（1個）、しょうゆ（大さじ2）、みりん（大さじ2）、砂糖（大さじ1）、サラダ油（大さじ1）\n    手順：\n    1. 肉を薄く広げ、ナスと玉ねぎを巻いて巻き終わりを下にしておく。\n    2. フライパンにサラダ油を熱し、肉巻きを焼く。\n    3. しょうゆ、みりん、砂糖を加えて煮詰める。\n    4. 火を止めて完成。"

    # 各レシピを分割してリストに格納
    recipes = answer.split('\n\n')

    # 各レシピの名前、材料、手順を辞書に格納
    recipe_details = []
    for recipe in recipes:
        lines = recipe.split('\n')
        recipe_name = lines[0].strip('・')

        ingredients_start_index = None
        for index, item in enumerate(lines):
            if '材料' in item:
                ingredients_start_index = index
                break
        instructions_start_index = None
        for index, item in enumerate(lines):
            if '手順' in item:
                instructions_start_index = index
                break

        ingredients = lines[ingredients_start_index : instructions_start_index] 
        procedure = lines[instructions_start_index:]

        ingredients = '\n'.join(line.strip() for line in ingredients)
        procedure = '\n'.join(line.strip() for line in procedure)

        recipe_details.append({
            'name': recipe_name,
            'ingredients': ingredients,
            'procedure': procedure
        })

    # # 辞書を出力
    # for detail in recipe_details:
    #     print(f"Recipe Name: {detail['name']}")
    #     print(f"ingredients: {detail['ingredients']}")
    #     print(f"procedure: {detail['procedure']}")

    return recipe_details

# デバッグ用
if __name__ == "__main__":
    prompt = f"りんご、はちみつ、バナナを使った3つのレシピの料理名と材料と手順を挙げてください。材料の分量を必ず教えてください。\
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

    recipes = chatgpt(prompt)
    # print(recipes)

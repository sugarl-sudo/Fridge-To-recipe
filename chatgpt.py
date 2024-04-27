import os
import requests
from dotenv import load_dotenv

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

    # エンドポイントへのPOSTリクエストのボディ
    body = f'''
    {{
        "model": "gpt-3.5-turbo",
        "messages": [
            {{"role": "user", "content": "{prompt}"}}
        ]
    }}
    '''

    # エンドポイントからのPOSTレスポンス
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=header, data=body.encode('utf_8'))
    # print(f"POSTレスポンスのJSONデータ: {response.text}")

    # デシリアライズ
    rj = response.json()
    # print('rj keys', rj.keys())

    answer = rj["choices"][0]["message"]["content"]
    # print(answer)

    # 各レシピを分割してリストに格納
    recipes = answer.split('\n\n')

    # 各レシピの名前、材料、手順を辞書に格納
    recipe_details = []
    for recipe in recipes:
        print(f"recipe = {recipe}")
        lines = recipe.split('\n')
        print(f"lines = {lines}")
        recipe_name = lines[0].strip('・')
        ingredients = lines[1].strip('    材料: ')
        procedure = '\n'.join(line.strip() for line in lines[2:])  # 手順の部分
        recipe_details.append({
            'name': recipe_name,
            'ingredients': ingredients,
            'procedure': procedure
        })

    # 辞書を出力
    print(f"Recipe: {recipe_details}")
    for detail in recipe_details:
        print(f"Recipe Name: {detail['name']}")
        print(f"Ingredients: {detail['ingredients']}")
        print(f"Procedure: {detail['procedure']}\n")

    # ChatGPTの回答を返す
    return rj["choices"][0]["message"]["content"]

# デバッグ用
if __name__ == "__main__":
    prompt = f"はちみつ、りんご、バナナを使った3つのレシピの料理名と材料と手順を挙げてください。材料の分量を必ず教えてください\
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
    # prompt = f"卵、鶏肉、七味を使ったレシピの料理名を3つ挙げてください。このとき、番号は表示しないでください。\
    #     出力のフォーマット：\
    #         料理名\
    #         料理名\
    #         料理名\
    #         "

    answer = chatgpt(prompt)
    # print(answer)

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
    response = requests.post("https://api.openai.com/v1/chat/completions", headers = header, data = body.encode('utf_8'))
    # print(f"POSTレスポンスのJSONデータ: {response.text}")

    # デシリアライズ
    rj = response.json()

    # ChatGPTの回答を返す
    return rj["choices"][0]["message"]["content"]

# デバッグ用
if __name__ == "__main__":
    prompt = f"卵、鶏肉、七味を使った3つのレシピの料理名と材料と手順を挙げてください。材料は分量についても教えてください\
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
    print(answer)
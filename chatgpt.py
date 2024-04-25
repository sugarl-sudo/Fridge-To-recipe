from datetime import datetime

# プロンプトからレシピを提案する関数
def chatgpt(prompt):
    answer = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") +'ChatGPTの回答'
    return answer
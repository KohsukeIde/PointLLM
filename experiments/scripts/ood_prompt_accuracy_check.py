import json
import os

def calculate_accuracy(file_path):
    """
    指定されたJSONファイルから精度を計算する
    
    Args:
        file_path: 判定結果のJSONファイルのパス
    
    Returns:
        float: 精度（正解率）
    """
    # JSONファイルを読み込む
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data["results"]  # JSONの構造に合わせて修正
    
    # 正解数と総数をカウント
    correct = 0
    total = len(results)
    
    for item in results:
        # internvl_verdictがCorrectであれば正解
        if item["internvl_verdict"] == "Correct":
            correct += 1
    
    # 精度を計算
    accuracy = correct / total if total > 0 else 0
    
    return accuracy, correct, total

# 各プロンプト(0-6)について個別に精度を計算
base_dir = "/groups/gag51404/ide/PointLLM/judge_results/"

# 各プロンプトについて明示的に計算
for i in range(7):  # 0から6まで
    file_path = f"{base_dir}ModelNet_OOD_prompt{i}_judged.json"
    if os.path.exists(file_path):
        try:
            accuracy, correct, total = calculate_accuracy(file_path)
            print(f"Prompt {i} の精度: {accuracy:.4f} ({correct}/{total}, {int(accuracy*100)}%)")
        except Exception as e:
            print(f"Prompt {i} の処理中にエラーが発生: {e}")
    else:
        print(f"Prompt {i} のファイルが見つかりません: {file_path}")
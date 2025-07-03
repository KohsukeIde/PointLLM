import json
import os
from pathlib import Path
import tqdm

# 対象となるID
target_ids = [
    "7dbb372a-75d0-4e3d-9954-06240f8bb8ad",
    "7f102cd0-803b-4aeb-b2ef-b14b241cbc5a",
    "803224dc-d327-4f97-b73a-943c8bad5d41",
    "ffed9e6c-5e6d-49aa-ba90-83927369ff47"
]

# 元のJSONファイルがあるディレクトリ
source_dir = "/groups/gag51404/ide/PointLLM/data/3d-grand-data/3d-grand-text-annotation/data/3D-FRONT/text_annotation"

# 抽出したJSONを保存するディレクトリ
output_dir = "/groups/gag51404/ide/PointLLM/data/3d-grand-data/data/sample/3D-Front/text_annotation"
os.makedirs(output_dir, exist_ok=True)

# 各JSONファイルを処理
for json_file in tqdm.tqdm(Path(source_dir).glob("*.json")):
    file_name = json_file.name
    print(f"処理中: {file_name}")
    
    try:
        # ファイル読み込み
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 抽出されたデータを格納する変数
        extracted_data = None
        
        # データ構造を推測して抽出（リスト形式または辞書形式）
        if isinstance(data, list):
            # リスト形式 [{"scene_id": "xxx", ...}, ...]
            # scene_idのキー名を推測（最初の要素から）
            if len(data) > 0:
                sample_keys = list(data[0].keys())
                id_key = next((k for k in sample_keys if "id" in k.lower() or "scene" in k.lower()), None)
                
                if id_key:
                    extracted_data = [item for item in data if any(target_id in str(item.get(id_key)) for target_id in target_ids)]
                else:
                    # IDキーが見つからない場合、全フィールドをチェック
                    extracted_data = []
                    for item in data:
                        for key, value in item.items():
                            if isinstance(value, str) and any(target_id in value for target_id in target_ids):
                                extracted_data.append(item)
                                break
        
        elif isinstance(data, dict):
            # 辞書形式 {"xxx": {...}, ...} またはネストされた構造
            extracted_data = {}
            
            # トップレベルのキーをチェック
            for key, value in data.items():
                # キーが直接ターゲットIDを含む場合（完全一致または部分一致）
                if any(target_id in key for target_id in target_ids):
                    extracted_data[key] = value
                # 値が辞書で、scene_idフィールドがターゲットIDに一致する場合
                elif isinstance(value, dict) and any(
                    target_id in str(value.get(k)) for target_id in target_ids 
                    for k in value.keys() if isinstance(k, str) and "id" in k.lower()
                ):
                    extracted_data[key] = value
        
        # 抽出されたデータがあれば保存
        if extracted_data and ((isinstance(extracted_data, list) and len(extracted_data) > 0) or 
                              (isinstance(extracted_data, dict) and len(extracted_data) > 0)):
            output_file = os.path.join(output_dir, f"sample_{file_name}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            print(f"保存完了: {output_file}、データ件数: {len(extracted_data)}")
        else:
            print(f"対象IDのデータが見つかりませんでした: {file_name}")
            
    except Exception as e:
        print(f"エラー ({file_name}): {e}")

print("すべてのファイルの処理が完了しました。")
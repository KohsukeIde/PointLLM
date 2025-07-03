from huggingface_hub import snapshot_download
import os

# 保存先を絶対パスで指定
local_dir = "/groups/gag51404/ide/PointLLM/data/3d-grand-data/data/sample"

# 保存先ディレクトリが存在することを確認
os.makedirs(local_dir, exist_ok=True)

# テキストアノテーションデータをダウンロード
snapshot_download(
    repo_id="sled-umich/3D-GRAND",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["data/3D-FRONT/text_annotation/**"],
    token=os.environ.get("HF_TOKEN")  # 環境変数からトークンを取得
)
print(f"データを {local_dir} にダウンロードしました")
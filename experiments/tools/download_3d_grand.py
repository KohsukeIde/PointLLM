from huggingface_hub import snapshot_download
import os

# サンプルデータのみをダウンロード
local_dir = "3d-grand-data"
snapshot_download(
    repo_id="sled-umich/3D-GRAND",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["data/sample/**"],
    token=os.environ.get("HF_TOKEN")  # 環境変数からトークンを取得
)
print(f"データを {local_dir} にダウンロードしました")

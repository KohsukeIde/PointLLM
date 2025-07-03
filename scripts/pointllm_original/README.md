# PointLLM Original Scripts - 元PointLLM訓練スクリプト（複数点群対応版）

オリジナルのPointLLM論文で使用された標準的な2段階訓練スクリプトに、複数点群対応機能を追加した拡張版です。

## 🚀 **新機能: 複数点群対応**

このバージョンでは、単一の対話で複数の点群を同時に処理できるように拡張されています：
- `<pc_1>`, `<pc_2>`, `<pc_3>` などの動的プレースホルダー対応
- 複数点群間の比較・関係性理解
- 既存の単一点群機能との完全な後方互換性

## 📋 スクリプト一覧

### 🎓 **標準訓練**
- **`PointLLM_train_stage1.sh`**
  - Stage 1: Point Encoder の事前訓練
  - PointBERTベースの点群特徴抽出器を訓練
  - 使用データ: Cap3D point cloud データ

- **`PointLLM_train_stage2.sh`**
  - Stage 2: Language Model との統合訓練（複数点群対応）
  - 事前訓練済みLLaMA-7Bとpoint encoderを統合
  - 使用データ: `complex_instruction_stage2_multi_pc_70K.json`

- **`PointLLM_train_stage2_debug.sh`**
  - Stage 2 デバッグ版（小規模テスト用）
  - 1GPU対応、少ないサンプル数でのテスト

### 🔧 **デバッグツール**
- **`debug_tools/test_dataloader_multi_pc.py`** - 複数点群データローダーのテスト
- **`debug_tools/test_real_dataloader.py`** - 実データでのテスト

## 🚀 使用方法

### 基本的な使用方法
```bash
# Stage 1 実行
cd /groups/gag51404/ide/PointLLM/scripts/pointllm_original
./PointLLM_train_stage1.sh

# Stage 2 実行（複数点群対応版）
./PointLLM_train_stage2.sh

# Stage 2 デバッグ実行
./PointLLM_train_stage2_debug.sh
```

### 計算ノードでの実行（推奨）
```bash
# インタラクティブセッション取得
qsub -I -P gag51402 -q rt_HG -l select=1 -l walltime=2:00:00

# 環境アクティベート
cd /groups/gag51404/ide/PointLLM
source ../venv/pointllm/bin/activate

# トレーニング実行
bash scripts/pointllm_original/PointLLM_train_stage2_debug.sh
```

## ⚙️ 設定内容

### Stage 1 設定（従来通り）
- **Epochs**: 60
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW
- **データセット**: Cap3D (660K samples)

### Stage 2 設定（複数点群対応）
- **Epochs**: 3
- **Batch Size**: 8（メイン）/ 2（デバッグ）
- **Learning Rate**: 2e-5
- **データセット**: `complex_instruction_stage2_multi_pc_70K.json` (70K samples)
- **対話タイプ**: `simple_description`

## 🔧 実装した変更内容

### 1. データローダー拡張 (`pointllm/data/object_point_dataset.py`)

#### 変更点
```python
# 複数点群フィールドの対応
if 'object_ids' in sources[0]:  # 新しい複数点群形式
    object_ids = sources[0]['object_ids']
    for obj_id in object_ids:
        pc = self._load_point_cloud(obj_id)
        all_point_clouds.append(torch.from_numpy(pc.astype(np.float32)))
    point_cloud = all_point_clouds  # リスト形式

elif 'object_id' in sources[0]:  # 従来の単一点群形式
    object_id = sources[0]['object_id']
    pc = self._load_point_cloud(object_id)
    point_cloud = torch.from_numpy(pc.astype(np.float32))  # テンソル形式
```

#### ファイルパス修正
```python
def _load_objaverse_point_cloud(self, object_id):
    filename = f"{object_id}_{self.pointnum}.npy"
    # 修正: 正しいサブディレクトリを指定
    point_cloud = np.load(os.path.join(self.data_path, f"{self.pointnum}_npy", filename))
```

#### 動的プレースホルダー置換
```python
# <pc_1>, <pc_2>, <pc_3> などを動的に検出・置換
conversation_text = sources[0][0]['value']
pc_placeholders = re.findall(r'<pc_\d+>', conversation_text)
for placeholder in set(pc_placeholders):
    conversation_text = conversation_text.replace(placeholder, replace_token)
```

### 2. Data Collator修正 (`pointllm/data/utils.py`)

#### 変更点
```python
if 'point_clouds' in instances[0]:
    point_clouds = [instance['point_clouds'] for instance in instances]
    
    if all(x is not None for x in point_clouds):
        # 複数点群（リスト）と単一点群（テンソル）の両対応
        if isinstance(point_clouds[0], list):
            batch['point_clouds'] = point_clouds  # ネストしたリストとして保持
        elif hasattr(point_clouds[0], 'shape') and all(hasattr(x, 'shape') and x.shape == point_clouds[0].shape for x in point_clouds):
            batch['point_clouds'] = torch.stack(point_clouds)
        else:
            batch['point_clouds'] = point_clouds
```

### 3. モデル修正 (`pointllm/model/pointllm.py`)

#### 変更点
```python
if type(point_clouds) is list:
    # バッチごとの複数点群処理
    point_features = []
    for batch_clouds in point_clouds:  # バッチ内の各サンプル
        if isinstance(batch_clouds, list):
            # 複数点群の場合: [pc1_tensor, pc2_tensor, ...]
            batch_features = []
            for point_cloud in batch_clouds:
                point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
                batch_features.append(point_feature)
            point_features.append(batch_features)
        else:
            # 単一点群の場合: pc_tensor
            point_feature = self.point_backbone(batch_clouds.unsqueeze(0))[0]
            point_features.append(point_feature)
```

### 4. 必要なインポート追加
```python
# pointllm/data/object_point_dataset.py
import re  # 動的プレースホルダー検出用
```

## 📊 想定される出力

### Stage 1（従来通り）
- `outputs/PointLLM_train_stage1/` - Point Encoder チェックポイント
- `wandb/` - 訓練ログ

### Stage 2（複数点群対応）
- `outputs/PointLLM_train_stage2/` - 統合モデル チェックポイント
- `wandb/` - 訓練ログ
- **成功例**: Loss 7.9689 → 7.6922 → 3.6882 → 2.2254 (順調に減少)

## ⚠️ 注意事項

### システム要件
- **GPU要件**: H200 80GB以上推奨（1GPU対応版も提供）
- **データ要件**: 
  - Objaverse点群データ（`data/objaverse_data/8192_npy/`）
  - 複数点群データセット（`complex_instruction_stage2_multi_pc_70K.json`）

### トラブルシューティング
1. **ファイルパスエラー**: `8192_npy`サブディレクトリが正しく設定されているか確認
2. **GPU数エラー**: 利用可能なGPU数に合わせてスクリプトを調整
3. **conversation_typeエラー**: データセットの`conversation_type`が`simple_description`であることを確認

### データセット形式
```json
{
  "object_ids": ["id1", "id2", "id3"],  // 複数点群
  "conversations": [
    {
      "from": "human",
      "value": "Compare <pc_1>, <pc_2>, and <pc_3>. What are the differences?"
    },
    {
      "from": "gpt", 
      "value": "The first object <pc_1> is..."
    }
  ]
}
```

## 🔄 後方互換性

この拡張版は既存の単一点群機能と完全に互換性があります：
- 従来の`object_id`フィールドも引き続きサポート
- 既存の`<point>`プレースホルダーも動作
- Stage 1訓練は変更なし

## 🔗 関連ファイル

### 変更したファイル
- `pointllm/data/object_point_dataset.py` - データローダー拡張
- `pointllm/data/utils.py` - data collator修正  
- `pointllm/model/pointllm.py` - モデル修正
- `scripts/pointllm_original/PointLLM_train_stage2_debug.sh` - デバッグスクリプト

### 新規作成したファイル
- `debug_tools/test_dataloader_multi_pc.py` - データローダーテスト
- `debug_tools/test_real_dataloader.py` - 実データテスト

### 設定ファイル
- `data/anno_data/complex_instruction_stage2_multi_pc_70K.json` - 複数点群データセット

## 🎯 使用例

### 複数点群での対話例
```
Human: What are the differences between <pc_1> and <pc_2>?
AI: The first object <pc_1> is a red chair with four legs, while <pc_2> is a blue table with a rectangular surface... 
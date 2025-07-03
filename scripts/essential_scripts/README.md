# Essential Scripts - 核心スクリプト群

序数参照学習プロジェクトで最重要なスクリプト群です。

## 📋 スクリプト一覧

### 🎯 **メインパイプライン**
- **`run_lora_training.sh`** ⭐ **最重要**
  - 最高速設定でのLoRA学習 (18分, 100GB)
  - batch=4, grad_accum=2, workers=8の最適化設定
  - GPU計算ノードで実行: `qsub -I -P gag51402 -q rt_HG -l select=1 -l walltime=2:00:00`

- **`test_ordinal_training_pipeline_cap3d.sh`** ⭐ **重要**
  - データ生成 → LoRA訓練 → 評価までの完全パイプライン
  - 初回セットアップ時に使用

### 📊 **評価・比較**
- **`compare_evaluation_results.py`** ⭐ **最重要**
  - ベースモデル vs LoRAモデルの詳細比較
  - 精度改善・グラフ生成・統計レポート作成
  - 使用例: `python compare_evaluation_results.py --base_results eval_base.json --lora_results eval_lora.json`

### 🔒 **安全性チェック**
- **`lora_safety_demo.py`** ⭐ **重要**
  - LoRA適用時の元モデル重みが変更されないことを実証
  - 使用例: `python lora_safety_demo.py`

- **`verify_base_model_integrity.py`** ⭐ **重要**
  - 元モデルの完整性チェック
  - LoRA訓練後に元モデルが破損していないことを確認

## 🚀 クイックスタート

```bash
# 1. LoRA学習実行 (最高速設定)
cd /groups/gag51404/ide/PointLLM/scripts/essential_scripts
./run_lora_training.sh

# 2. 結果比較
python compare_evaluation_results.py \
    --base_results ../../evaluation/baseline/ \
    --lora_results ../../evaluation/lora_trained/ \
    --output_dir ../../evaluation/comparison

# 3. 安全性チェック
python lora_safety_demo.py
python verify_base_model_integrity.py
```

## ⚠️ 注意事項

- **GPU必須**: パイプラインスクリプトは必ずGPU計算ノードで実行
- **メモリ要件**: 最低32GB VRAM推奨
- **実行時間**: 完全パイプラインは約3-6時間

## 📝 実行ログ例

```
========================================
PointLLM Cap3D Ordinal Training Pipeline
========================================
Step 1: Installing Dependencies ✅
Step 2: Generating Cap3D Ordinal Dataset ✅
Step 3: Baseline Evaluation ✅
Step 4: LoRA Training ✅
Step 5: Post-Training Evaluation ✅
Step 6: Results Summary ✅
========================================
``` 
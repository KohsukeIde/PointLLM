# PointLLM Dependencies & Bug Fix Summary
# PointLLM 依存関係とバグ修正履歴

このドキュメントでは、PointLLMプロジェクトで発生した主要な依存関係の問題とバグ修正について記録します。

## 🐛 主要なバグと修正内容

### 1. APEX amp モジュールのインポートエラー

**問題:**
```bash
ImportError: cannot import name 'amp' from 'apex' 
RuntimeError: Failed to import transformers.trainer because of the following error
```

**原因:**
- APEXがCUDA拡張なしでインストールされており、`amp`モジュールが含まれていない
- Transformersライブラリが内部でAPEXのampモジュールを必要としている

**修正内容:**
1. **一時的な解決策:** Mock ampモジュールの作成
   - ファイル: `create_mock_amp.py`
   - APEXのampモジュールの基本機能をエミュレート
   ```python
   amp_module.initialize = lambda model, optimizers=None, **kwargs: (model, optimizers)
   amp_module.scale_loss = lambda loss, optimizer: loss
   ```

2. **根本的解決策:** APEX CUDA拡張付きインストール
   - 要求: CUDA環境でのソースからのビルド
   - コマンド: `pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/`

### 2. NetworkX バックエンド警告

**問題:**
```bash
RuntimeWarning: networkx backend defined more than once: nx-loopback
```

**原因:**
- NetworkXライブラリで重複したバックエンド定義

**影響:**
- 機能に影響はないが、ログが冗長になる

**対応:**
- 現在は警告として受容（機能的な問題なし）

### 3. Timm レイヤーインポート警告

**問題:**
```bash
FutureWarning: Importing from timm.models.layers is deprecated
```

**原因:**
- Timmライブラリのレイヤーインポートパスが非推奨

**対応:**
- 新しいインポートパス（`timm.layers`）への移行が推奨

## 📋 削除されたファイル履歴

以下のスクリプトファイルが整理により削除されました：

### 削除されたファイル:
- `PointLLM_train_stage1_fixed.sh` - バグ修正版の訓練スクリプト
- `PointLLM_train_stage1_single_gpu.sh` - シングルGPU対応版
- `PointLLM_train_stage1_simple.sh` - 簡略化版

### 削除理由:
- 標準的な訓練スクリプト（`PointLLM_train_stage1.sh`、`PointLLM_train_stage2.sh`）に統合
- コードベースの整理とメンテナンス性向上

## ⚙️ 現在の解決済み問題

### 1. 依存関係の互換性
- **Status:** ✅ 解決済み
- **Solution:** Mock ampモジュールによる一時的な回避

### 2. 環境設定
- **Status:** ⚠️ 要監視
- **Note:** 計算ノード使用の必須ルール適用中

## 🔧 推奨される今後の対応

### 短期的対応
1. **APEX完全インストール:** CUDA拡張付きの完全なAPEXインストール
2. **依存関係固定:** requirements.txtでの厳密なバージョン管理

### 長期的対応
1. **代替ライブラリ検討:** APEXに依存しない混合精度訓練の実装
2. **コンテナ化:** Docker/Singularityによる一貫した環境構築

## 📊 環境情報

### システム環境
- **OS:** Linux 5.14.0-427.13.1.el9_4.x86_64
- **Python:** 3.9
- **計算環境:** HPC クラスター（要計算ノード使用）

### 重要な制約
- **必須:** 計算ノードでの実行（ログインノード使用禁止）
- **コマンド:** `qsub -I -P gag51402 -q rt_HG -l select=1 -l walltime=2:00:00`

## 🔗 関連ファイル

- `create_mock_amp.py` - APEX amp問題の一時回避策
- `pointllm_training.log` - 訓練時のエラーログ
- `scripts/pointllm_original/README.md` - 標準訓練スクリプトの説明

---

**最終更新:** このドキュメント作成時点
**維持管理者:** PointLLMプロジェクトチーム 
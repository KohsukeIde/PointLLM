# Archive Scripts - アーカイブスクリプト群

開発過程で作成されたが、現在は使用されていない古いスクリプトのアーカイブです。

## 📋 アーカイブされたスクリプト

### 🗂️ **開発初期版**
- **`test_ordinal_training_pipeline.sh`**
  - 初期のObjaverseベース序数訓練パイプライン
  - **置き換え**: `../essential_scripts/test_ordinal_training_pipeline_cap3d.sh`
  - **非推奨理由**: Cap3Dベース版でより安定・高精度

- **`run_ordinal_training.sh`**
  - 単独のLoRA訓練スクリプト
  - **置き換え**: メインパイプラインスクリプトに統合済み
  - **非推奨理由**: データ生成から評価までの一貫性確保のため

- **`generate_ordinal_dataset.sh`**
  - 初期のデータセット生成スクリプト
  - **置き換え**: `../../ordinal_reference_project/dataset_generation/`
  - **非推奨理由**: より高品質なデータ生成ロジックの実装

## ⚠️ 重要な注意

**これらのスクリプトは実行しないでください**

1. **データ品質**: 古いデータ生成ロジックは不正確な序数参照を生成
2. **互換性**: 現在のプロジェクト構造と互換性なし
3. **性能**: 最新のCap3Dベース実装より大幅に劣る

## 🔄 移行ガイド

### 古いスクリプト → 新しいスクリプト

```bash
# ❌ 古い方法
./archive/test_ordinal_training_pipeline.sh

# ✅ 新しい方法
./essential_scripts/test_ordinal_training_pipeline_cap3d.sh
```

```bash
# ❌ 古い方法
./archive/generate_ordinal_dataset.sh

# ✅ 新しい方法
cd ../../ordinal_reference_project/dataset_generation/
python ordinal_dataset_generator_cap3d.py
```

## 📚 歴史的価値

これらのスクリプトは以下の目的で保存されています：

1. **開発履歴**: プロジェクトの進化過程の記録
2. **比較研究**: 古い手法との性能比較
3. **参考実装**: 特定のロジックの参考資料

## 🗑️ 削除予定

以下の条件が満たされた場合、これらのファイルは削除される可能性があります：

- [ ] 現在の実装が十分に安定
- [ ] 性能評価が完了
- [ ] 開発チーム内での合意

## 🔗 現在の推奨スクリプト

代わりに以下を使用してください：

- **メインパイプライン**: `../essential_scripts/test_ordinal_training_pipeline_cap3d.sh`
- **データ生成**: `../../ordinal_reference_project/dataset_generation/`
- **評価比較**: `../essential_scripts/compare_evaluation_results.py` 
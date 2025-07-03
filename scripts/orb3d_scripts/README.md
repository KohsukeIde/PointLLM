# ORB3D Scripts - ORB3D専用スクリプト群

ORB3D Shape Mating タスク評価用のスクリプト群です。

## 📋 スクリプト一覧

### 🧪 **評価実行**
- **`eval_orb3d_mating.sh`**
  - ORB3D shape mating タスクの評価を実行
  - Binary classification と Multi-choice classification の両方に対応
  - 使用例: `./eval_orb3d_mating.sh`

### 📊 **結果分析**
- **`summarize_orb3d_results.py`**
  - ORB3D評価結果の要約とグラフ生成
  - 切り口タイプ別（planar/parabolic）の精度分析
  - プロンプト別の性能比較レポート作成

## 🚀 使用方法

```bash
# 1. ORB3D評価実行
cd /groups/gag51404/ide/PointLLM/scripts/orb3d_scripts
./eval_orb3d_mating.sh

# 2. 結果要約
python summarize_orb3d_results.py \
    --results_dir ../../evaluation/ORB3D_mating \
    --output_dir ../../evaluation/ORB3D_summary
```

## 📁 出力ファイル

### 評価結果
- `ORB3D_binary_mating_prompt{0-5}.json` - Binary classification 結果
- `ORB3D_multi_choice_mating_prompt{0-5}.json` - Multi-choice classification 結果

### 要約レポート
- `orb3d_mating_summary.txt` - テキスト要約
- `accuracy_by_prompt.png` - プロンプト別精度グラフ
- `accuracy_by_cut_type.png` - 切り口タイプ別精度グラフ

## 📈 分析項目

1. **プロンプト別精度**: 6種類のプロンプトでの性能比較
2. **切り口タイプ別**: Planar vs Parabolic カットの難易度分析
3. **タスク種別**: Binary vs Multi-choice の性能差
4. **全体統計**: 平均精度、サンプル数、信頼区間

## ⚠️ 前提条件

- ORB3D データセットが適切に配置されていること
- 評価用モデル（PointLLM）が読み込み可能であること
- matplotlib, pandas ライブラリがインストール済みであること

## 🔗 関連ドキュメント

- `../../ORB3D_MATING_EVALUATION.md` - ORB3D評価の詳細説明
- `../../orb3d/` - ORB3Dデータセット格納場所 
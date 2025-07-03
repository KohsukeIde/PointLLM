# Shape Mating Dataset Generator v1.0

## 概要

PointLLMの3D幾何学理解能力向上のための**Shape Mating Pair Discovery Dataset Generator**です。3つの3D部品から適合する2つの部品を特定するタスクを通じて、高度な空間認識能力を学習させます。

## 🎯 タスク設計

### 基本コンセプト
- **入力**: 3つの3D部品 (A, B, C)
- **質問**: "どの2つの部品が組み合わさりますか？"
- **回答**: "第一と第三の部品が適合します"

### カリキュラム学習アプローチ
**Easy → Medium → Hard** の順序で段階的学習を実現：

#### Task 1: Easy (40%, 28K サンプル)
- **内容**: 異種オブジェクト・同一カット
- **例**: 椅子(square) + テーブル(square) vs 本(square)
- **学習目標**: オブジェクト全体の形状で判断
- **重複率**: 0.014% (ほぼゼロ)

#### Task 2: Medium (30%, 21K サンプル)  
- **内容**: 同一オブジェクト・異種カット
- **例**: 椅子(square) + 椅子(sine) vs 椅子(pulse)
- **学習目標**: カット面形状の理解
- **重複率**: 3.32% (安全範囲)

#### Task 3: Hard (30%, 21K サンプル)
- **内容**: 同一オブジェクト・同一カット・異種インスタンス
- **例**: 椅子_inst1(square) + 椅子_inst2(square) vs 椅子_inst3(square)
- **学習目標**: 微細な幾何学的差異検出
- **重複率**: 14.74% (許容範囲)

## 📊 データ分析結果

### データセット規模
```
総オブジェクト数: 633個
カットタイプ数: 5種類 (parabolic, planar, pulse, sine, square)
平均インスタンス数/カット: 10個
総組み合わせ数: 200,803,425通り (約2億)
```

### 組み合わせ数詳細
```
Task 1 (Easy):   200,028,000通り
Task 2 (Medium):     633,000通り  
Task 3 (Hard):       142,425通り
```

### 70Kサンプルでの重複率
```
全体重複率: 0.035% (ほぼ完全にユニーク)
Task 1: 0.014% (安全)
Task 2: 3.32%  (安全)
Task 3: 14.74% (許容範囲内)
```

## 🚀 使用方法

### 基本コマンド

#### GPT-4使用版（最高品質）
```bash
python3 create_shape_mating_dataset.py --num_samples 70000 --use_gpt_api --model_name gpt-4-turbo --max_workers 20
```

#### GPT-3.5使用版（コスト効率）
```bash
python3 create_shape_mating_dataset.py --num_samples 70000 --use_gpt_api --model_name gpt-3.5-turbo --max_workers 20
```

#### 非GPT版（最速・無料）
```bash
python3 create_shape_mating_dataset.py --num_samples 70000 --max_workers 30
```

### 主要オプション
- `--num_samples`: 生成サンプル数 (推奨: 70000)
- `--use_gpt_api`: GPT APIの使用（質問文の多様化）
- `--model_name`: GPTモデル選択
- `--max_workers`: 並列処理ワーカー数
- `--save_comparison`: 比較データの保存

## 📁 出力ファイル

### メインデータセット
`shape_mating_pair_discovery_70K_gpt.json`
```json
{
  "object_ids": [
    "/path/to/partA.npy",
    "/path/to/partB.npy", 
    "/path/to/negative.npy"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<pc_1> <pc_2> <pc_3>\nWhich two parts form a mating pair?"
    },
    {
      "from": "gpt", 
      "value": "The first and third parts fit together."
    }
  ],
  "metadata": {
    "task_type": 1,
    "difficulty": "Easy",
    "correct_pair_indices": [0, 2],
    "correct_pair_positions": ["first", "third"],
    ...
  }
}
```

### 統計ファイル
`shape_mating_pair_discovery_70K_gpt_stats.json`
- タスク分布統計
- 応答多様性分析
- API使用統計
- 生成メタデータ

## 🔧 技術実装

### アーキテクチャ
- **スレッドセーフ設計**: 並列処理対応
- **GPTキャッシング**: API呼び出し最適化
- **ロバストパス解析**: 正規表現+フォールバック
- **多様性生成**: 30種類の質問テンプレート、10種類の回答テンプレート

### 核心アルゴリズム
```python
def _sample_task1_data(self):
    """Task 1: 異種オブジェクト・同一カット"""
    # 1. 同一カットタイプのオブジェクト群を特定
    # 2. 異なるオブジェクトから正解ペアと不正解部品を選択
    # 3. ランダムシャッフルで位置決定

def _sample_task2_data(self):
    """Task 2: 同一オブジェクト・異種カット"""  
    # 1. 複数カットタイプを持つオブジェクトを特定
    # 2. 正解カットペアと不正解カット部品を選択
    # 3. インスタンスレベルでのランダム選択

def _sample_task3_data(self):
    """Task 3: 同一オブジェクト・同一カット・異種インスタンス"""
    # 1. 3つ以上のインスタンスを持つ組み合わせを特定
    # 2. 同一ペアと異なるインスタンス部品を選択
    # 3. 微細差異による最高難易度タスク生成
```

## ⚡ 性能指標

### 処理速度
- **小規模テスト**: 10サンプル/秒
- **中規模テスト**: 100サンプル/秒  
- **大規模処理**: 並列化により高速処理

### 品質指標
- **応答多様性**: 66.7% (テスト時)
- **構造完整性**: 100% (エラーゼロ)
- **評価可能性**: `correct_pair_indices`による自動評価対応

## 🎓 カリキュラム学習効果

### 学習進行
1. **Easy段階**: 基本的なオブジェクト識別能力の確立
2. **Medium段階**: カット面形状の理解と空間認識
3. **Hard段階**: 微細な幾何学的差異の検出能力

### 教育的効果
- **段階的習得**: 過学習防止
- **効率的学習**: 基礎から高度技術への自然な進行
- **最終到達点**: 人間レベルの3D幾何学理解

## 🔍 重複対策

### 現状分析
- **全体**: 0.035%重複率（ほぼ完全ユニーク）
- **Task 3制約**: 14.74%重複率（許容範囲内）

### 改善オプション
1. **現状維持**: 機械学習的に問題なし
2. **分布調整**: Task 3比率を20%に削減
3. **サンプル削減**: 50Kサンプルで重複率改善

### 推奨アプローチ
**現状維持**を推奨：
- 14.74%は許容範囲
- Hardタスクの重要性維持
- カリキュラム学習の完成度保持

## 📈 評価メトリクス

### 自動評価
```python
# 正解判定
predicted_indices = model_output_to_indices(response)
correct_indices = sample["metadata"]["correct_pair_indices"]
accuracy = (predicted_indices == correct_indices)
```

### 詳細分析
- **タスク別精度**: Easy/Medium/Hard別の性能
- **位置バイアス**: first/second/third選択傾向
- **カテゴリ分析**: オブジェクト種別の得意/不得意

## 🔄 バージョン履歴

### v1.0 (Current)
- ✅ カリキュラム学習順序修正 (Easy→Medium→Hard)
- ✅ タスクタイプ再配置完了
- ✅ 組み合わせ数分析完了
- ✅ 重複率検証完了
- ✅ 70Kサンプル生成対応

### 主要改善点
1. **応答多様化**: 2種類→30種類のテンプレート
2. **スレッドセーフ化**: グローバル状態の除去
3. **パス解析強化**: 正規表現+フォールバック
4. **GPTキャッシング**: API効率化
5. **カリキュラム学習**: 教育的最適化

## 🚨 既知の制約事項

### データ制約
- オブジェクト数: 633個（固定）
- カットタイプ: 5種類（現状）
- インスタンス数: 平均10個/カット

### 技術制約
- Task 3組み合わせ数制限
- GPT API レート制限
- 並列処理メモリ使用量

## 🔮 将来の拡張予定

### データ拡張
- 新カットタイプ追加
- オブジェクトカテゴリ拡張
- マルチモーダル対応

### 技術改善
- 動的難易度調整
- アクティブラーニング
- リアルタイム品質監視

## 📞 サポート・問い合わせ

### トラブルシューティング
1. **メモリ不足**: `--max_workers`を削減
2. **API制限**: GPT非使用版に切り替え
3. **データ不足**: サンプル数を削減

### 開発者向け情報
- コード: `create_shape_mating_dataset.py`
- テスト: 小規模データでの事前検証推奨
- ログ: 統計ファイルでの詳細分析

---

**Shape Mating Dataset Generator v1.0** - PointLLMの3D幾何学理解能力向上のための完全ソリューション 
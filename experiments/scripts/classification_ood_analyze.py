import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def extract_noise_params(filename):
    """ファイル名からノイズパラメータを抽出"""
    match = re.search(r'evaluation_(kr[\d\.]+)_(js[\d\.]+)_(sh\d+)_', filename)
    if match:
        kr = float(match.group(1).replace('kr', ''))
        js = float(match.group(2).replace('js', ''))
        sh = int(match.group(3).replace('sh', ''))
        return kr, js, sh
    return None, None, None

def parse_evaluation_file(filepath):
    """評価ファイルからデータを抽出"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 全体精度の抽出
    accuracy_match = re.search(r'全体精度: ([\d\.]+)', content)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None
    
    # クラス別詳細の抽出（より複雑な処理が必要）
    class_data = {}
    in_class_section = False
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "クラス別詳細レポート:" in line:
            in_class_section = True
            continue
        if in_class_section and line.strip() and "precision" not in line and "macro avg" not in line and "weighted avg" not in line:
            parts = line.strip().split()
            if len(parts) >= 4:  # クラス名とスコアがある行
                class_name = parts[0]
                try:
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1 = float(parts[3])
                    class_data[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}
                except (ValueError, IndexError):
                    continue
    
    return {
        'accuracy': accuracy,
        'class_data': class_data,
        'kr': kr,
        'js': js,
        'sh': sh
    }

# 結果ディレクトリのパス
results_dir = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/noise_ood/results"

# 全評価ファイルを処理
data = []
for filename in os.listdir(results_dir):
    if filename.startswith("evaluation_") and filename.endswith(".txt"):
        filepath = os.path.join(results_dir, filename)
        kr, js, sh = extract_noise_params(filename)
        if kr is not None:
            file_data = parse_evaluation_file(filepath)
            file_data.update({'kr': kr, 'js': js, 'sh': sh})
            data.append(file_data)

# データフレームに変換
df = pd.DataFrame(data)

# 1. ノイズパラメータと精度の関係
plt.figure(figsize=(12, 8))

# Keep Ratioの影響（jitter_stdごとにグループ化）
plt.subplot(2, 2, 1)
for js in df['js'].unique():
    subset = df[df['js'] == js]
    plt.plot(subset['kr'], subset['accuracy'], marker='o', label=f'js={js}')
plt.xlabel('Keep Ratio (kr)')
plt.ylabel('Accuracy')
plt.title('Impact of Keep Ratio on Accuracy')
plt.legend()

# Jitter Stdの影響（keep_ratioごとにグループ化）
plt.subplot(2, 2, 2)
for kr in df['kr'].unique():
    subset = df[df['kr'] == kr]
    plt.plot(subset['js'], subset['accuracy'], marker='o', label=f'kr={kr}')
plt.xlabel('Jitter Std (js)')
plt.ylabel('Accuracy')
plt.title('Impact of Jitter Std on Accuracy')
plt.legend()

# Shuffleの影響
plt.subplot(2, 2, 3)
sns.boxplot(x='sh', y='accuracy', data=df)
plt.xlabel('Shuffle (sh)')
plt.ylabel('Accuracy')
plt.title('Impact of Shuffle on Accuracy')

plt.tight_layout()
plt.savefig('noise_param_impact.png')
plt.close()

# 2. クラス別の影響分析
# 各クラスのF1スコアを抽出
class_f1 = defaultdict(list)
for entry in data:
    for class_name, scores in entry['class_data'].items():
        class_f1[class_name].append({
            'f1': scores['f1'],
            'kr': entry['kr'],
            'js': entry['js'],
            'sh': entry['sh']
        })

# トップ5の頑健なクラスとトップ5の脆弱なクラス
class_avg_f1 = {}
for class_name, scores in class_f1.items():
    class_avg_f1[class_name] = np.mean([s['f1'] for s in scores])

robust_classes = sorted(class_avg_f1.items(), key=lambda x: x[1], reverse=True)[:5]
vulnerable_classes = sorted(class_avg_f1.items(), key=lambda x: x[1])[:5]

print("最も頑健なクラス（F1スコア）:")
for cls, f1 in robust_classes:
    print(f"  {cls}: {f1:.4f}")

print("\n最も脆弱なクラス（F1スコア）:")
for cls, f1 in vulnerable_classes:
    print(f"  {cls}: {f1:.4f}")

# 詳細な分析結果をCSVファイルに保存
output_df = pd.DataFrame(data)
output_df.to_csv('noise_analysis_results.csv', index=False)

print("\n分析完了。結果はnoise_param_impact.pngとnoise_analysis_results.csvに保存されました。")
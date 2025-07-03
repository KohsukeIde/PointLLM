import json
import os
import torch
import numpy as np
import re
from sklearn.metrics import accuracy_score, classification_report
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime


def evaluate_with_clip(result_file, categories_file=None):
    """CLIPを使用してモデル出力と各カテゴリの意味的類似度を評価"""
    # CLIP モデルとプロセッサをロード
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 結果ファイルの読み込み
    print(f"Reading results from {result_file}...")
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    print(f"Found {len(results)} samples in results")
    
    # カテゴリ名の読み込み
    if categories_file and os.path.exists(categories_file):
        categories = [line.rstrip() for line in open(categories_file)]
    else:
        # ファイルがない場合、結果から一意なラベルを収集
        categories = list(set(sample["label_name"] for sample in results))
        categories.sort()
    
    print(f"Using {len(categories)} categories")
    
    # カテゴリのテキスト表現を作成
    category_texts = [f"This is a {category}" for category in categories]
    
    # カテゴリのテキスト埋め込みを事前計算
    with torch.no_grad():
        inputs = processor(text=category_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        category_embeddings = model.get_text_features(**inputs)
        category_embeddings = category_embeddings / category_embeddings.norm(dim=-1, keepdim=True)
    
    # 予測と正解ラベルの抽出
    true_labels = []
    pred_labels = []
    confidence_scores = []
    incorrect_samples = []
    
    # モデル出力をバッチで処理
    batch_size = 32
    for i in tqdm(range(0, len(results), batch_size)):
        batch = results[i:i+batch_size]
        
        # バッチの出力テキストを取得
        model_outputs = [sample["model_output"] for sample in batch]
        true_cats = [sample["label_name"] for sample in batch]
        
        # テキスト埋め込みを計算
        with torch.no_grad():
            inputs = processor(text=model_outputs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_embeddings = model.get_text_features(**inputs)
            output_embeddings = output_embeddings / output_embeddings.norm(dim=-1, keepdim=True)
        
        # 各出力とすべてのカテゴリとの類似度を計算
        similarities = output_embeddings @ category_embeddings.T
        
        # 最も類似度の高いカテゴリを予測として選択
        predicted_indices = similarities.argmax(dim=1).cpu().numpy()
        confidence = similarities.max(dim=1).values.cpu().numpy()
        
        for j, (true_cat, pred_idx, conf, output) in enumerate(zip(true_cats, predicted_indices, confidence, model_outputs)):
            sample_idx = i + j
            pred_cat = categories[pred_idx]
            
            true_labels.append(true_cat)
            pred_labels.append(pred_cat)
            confidence_scores.append(float(conf))
            
            if true_cat != pred_cat:
                incorrect_samples.append({
                    "id": results[sample_idx]["object_id"],
                    "true": true_cat,
                    "pred": pred_cat,
                    "confidence": float(conf),
                    "text": output[:100] + "..." 
                })
    
    # 分類精度の計算
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n精度: {accuracy:.4f} ({sum(1 for t, p in zip(true_labels, pred_labels) if t == p)}/{len(true_labels)})")
    
    # 詳細なレポート
    print("\n分類レポート:")
    class_report = classification_report(true_labels, pred_labels, target_names=categories, zero_division=0)
    print(class_report)
    
    # 信頼度の分布
    print(f"\n予測信頼度統計:")
    conf_mean = np.mean(confidence_scores)
    conf_min = np.min(confidence_scores)
    conf_max = np.max(confidence_scores)
    print(f"  平均: {conf_mean:.4f}")
    print(f"  最小: {conf_min:.4f}")
    print(f"  最大: {conf_max:.4f}")
    
    # 信頼度閾値別の精度
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    print("\n信頼度閾値別の精度:")
    threshold_results = []
    for threshold in thresholds:
        filtered_indices = [i for i, conf in enumerate(confidence_scores) if conf >= threshold]
        if filtered_indices:
            filtered_true = [true_labels[i] for i in filtered_indices]
            filtered_pred = [pred_labels[i] for i in filtered_indices]
            filtered_acc = accuracy_score(filtered_true, filtered_pred)
            print(f"  閾値 {threshold:.2f}: {filtered_acc:.4f} ({len(filtered_indices)}/{len(confidence_scores)} サンプル)")
            threshold_results.append((threshold, filtered_acc, len(filtered_indices), len(confidence_scores)))
    
    # 誤分類サンプルの分析（信頼度順）
    if incorrect_samples:
        print(f"\n誤分類サンプル (信頼度順): {len(incorrect_samples)}/{len(true_labels)}")
        sorted_incorrect = sorted(incorrect_samples, key=lambda x: x["confidence"], reverse=True)
        for i, sample in enumerate(sorted_incorrect[:5]):
            print(f"  サンプル {sample['id']} (真値: {sample['true']}, 予測: {sample['pred']}, 信頼度: {sample['confidence']:.4f}):")
            print(f"    {sample['text']}")
        if len(incorrect_samples) > 5:
            print(f"  ... 他 {len(incorrect_samples)-5} サンプル")
    
    # 結果をテキストファイルに保存
    noise_params = extract_noise_params(result_file)
    save_results_to_txt(
        accuracy, 
        class_report, 
        conf_mean, conf_min, conf_max,
        threshold_results,
        len(incorrect_samples),
        len(true_labels),
        noise_params
    )
    
    # 混同行列のヒートマップをセーブ
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # ノイズパラメータを抽出してファイル名に含める
        output_file = f'confusion_matrix_clip_{noise_params}.png'
        
        cm = confusion_matrix(true_labels, pred_labels, labels=categories)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Noise: {noise_params})')
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"\n混同行列を '{output_file}' に保存しました")
    except Exception as e:
        print(f"混同行列の保存でエラー: {e}")

    return incorrect_samples, categories, extract_noise_params(result_file)

def save_results_to_txt(accuracy, class_report, conf_mean, conf_min, conf_max, 
                        threshold_results, incorrect_count, total_count, noise_params):
    """評価結果をテキストファイルに保存"""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/evaluation_{noise_params}_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"=== PointLLM ノイズデータ評価結果 ===\n")
        f.write(f"日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ノイズパラメータ: {noise_params}\n\n")
        
        f.write(f"全体精度: {accuracy:.4f} ({total_count - incorrect_count}/{total_count})\n\n")
        
        f.write("信頼度統計:\n")
        f.write(f"  平均: {conf_mean:.4f}\n")
        f.write(f"  最小: {conf_min:.4f}\n")
        f.write(f"  最大: {conf_max:.4f}\n\n")
        
        f.write("信頼度閾値別の精度:\n")
        for threshold, acc, filtered_count, total in threshold_results:
            f.write(f"  閾値 {threshold:.2f}: {acc:.4f} ({filtered_count}/{total} サンプル)\n")
        f.write("\n")
        
        f.write(f"誤分類サンプル数: {incorrect_count}/{total_count}\n\n")
        
        f.write("クラス別詳細レポート:\n")
        f.write(class_report)
    
    print(f"\n評価結果を '{filename}' に保存しました")

def extract_noise_params(result_file):
    """結果ファイルのパスからノイズパラメータを抽出"""
    # 例: .../ModelNet_classification_kr0.30_js0.000_sh0_prompt0.json
    match = re.search(r'_(kr[\d\.]+_js[\d\.]+_sh\d+)', result_file)
    if match:
        return match.group(1)
    return "unknown_noise"

def render_misclassified_samples(incorrect_samples, data_path, output_dir, categories, noise_params, max_samples=10):
    """誤分類されたサンプルをレンダリングして画像として保存する"""
    print(f"\n誤分類サンプルの点群をレンダリングします（最大{max_samples}個）...")
    
    # 出力ディレクトリの作成
    output_dir = os.path.join(output_dir, noise_params)
    os.makedirs(output_dir, exist_ok=True)
    
    # データセットのロード
    print(f"点群データを読み込み中: {data_path}")
    with open(data_path, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)
    
    # 信頼度の高い順にソート
    sorted_samples = sorted(incorrect_samples, key=lambda x: x["confidence"], reverse=True)
    samples_to_render = sorted_samples[:max_samples]
    
    for i, sample in enumerate(tqdm(samples_to_render)):
        object_id = sample["id"]
        true_label = sample["true"]
        pred_label = sample["pred"]
        
        # 点群データの取得
        points = list_of_points[object_id]
        
        # 複数の視点からレンダリング
        fig = plt.figure(figsize=(15, 5))
        view_angles = [(30, 30), (30, 120), (30, 210)]
        view_names = ["front", "side", "back"]
        
        for j, (elev, azim) in enumerate(view_angles):
            ax = fig.add_subplot(1, 3, j+1, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.5)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"View: {view_names[j]}")
            ax.set_axis_off()
        
        plt.suptitle(f"ID: {object_id}, True: {true_label}, Pred: {pred_label} (Noise: {noise_params})")
        
        # 画像の保存
        filename = f"{output_dir}/misclassified_{i:03d}_id{object_id}_{true_label}_as_{pred_label}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)
        
    print(f"レンダリング完了: {len(samples_to_render)}個のサンプルを{output_dir}に保存しました")

def get_matching_noisy_data_path(result_file):
    """結果ファイルから対応するノイズデータのパスを生成"""
    # ノイズパラメータを抽出
    noise_params = extract_noise_params(result_file)
    
    # 対応するデータパスを生成
    base_dir = "/groups/gag51404/ide/PointLLM/data/modelnet40_data/ModelNet40_noisy"
    data_path = os.path.join(base_dir, noise_params, "modelnet40_test_8192pts_fps_noisy.dat")
    
    if not os.path.exists(data_path):
        print(f"警告: 点群データファイルが見つかりません: {data_path}")
    
    return data_path

def create_summary_report(result_files, output_file="noise_evaluation_summary.txt"):
    """複数のノイズパラメータ評価結果の要約レポートを作成"""
    # 各ノイズパラメータの結果を収集
    summary_data = []
    
    for result_file in result_files:
        noise_params = extract_noise_params(result_file)
        # 直近の評価結果ファイルを見つける
        result_files = [f for f in os.listdir("results") if f.startswith(f"evaluation_{noise_params}")]
        if result_files:
            # 最新のファイルを使用
            latest_file = sorted(result_files)[-1]
            with open(os.path.join("results", latest_file), 'r') as f:
                lines = f.readlines()
                # 精度の行を探す
                accuracy_line = [line for line in lines if line.startswith("全体精度:")]
                if accuracy_line:
                    accuracy = float(accuracy_line[0].split(":")[1].strip().split(" ")[0])
                    summary_data.append((noise_params, accuracy))
    
    # ノイズパラメータでソート
    summary_data.sort()
    
    # 要約レポートを作成
    with open(output_file, 'w') as f:
        f.write("=== ノイズ評価結果の要約 ===\n")
        f.write(f"作成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("ノイズパラメータ別の精度:\n")
        for noise_params, accuracy in summary_data:
            f.write(f"  {noise_params}: {accuracy:.4f}\n")
    
    print(f"\n要約レポートを '{output_file}' に保存しました")

if __name__ == "__main__":
    # ノイズデータディレクトリ
    noise_results_dir = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/noise_ood"
    categories_file = "/groups/gag51404/ide/PointLLM/modelnet_config/modelnet40_shape_names_modified.txt"
    output_base_dir = "noisy_misclassified_renders"
    
    # 結果ファイルのリストを取得
    result_files = [os.path.join(noise_results_dir, f) for f in os.listdir(noise_results_dir) 
                   if f.endswith('.json') and 'ModelNet_classification' in f]
    
    for result_file in result_files:
        print(f"\n処理中: {result_file}")
        
        # 対応する点群データのパスを取得
        data_path = get_matching_noisy_data_path(result_file)
        
        # CLIPによる評価
        incorrect_samples, categories, noise_params = evaluate_with_clip(result_file, categories_file)
        
        # 点群レンダリング
        render_misclassified_samples(incorrect_samples, data_path, output_base_dir, categories, noise_params, max_samples=20)
    
    # すべての評価が終わったら要約レポートを作成
    create_summary_report(result_files)
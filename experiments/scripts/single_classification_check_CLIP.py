import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def evaluate_with_clip(result_file, categories_file=None, output_dir=None):
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
    classification_rep = classification_report(true_labels, pred_labels, target_names=categories, zero_division=0)
    print("\n分類レポート:")
    print(classification_rep)
    
    # 信頼度の分布
    confidence_stats = {
        "mean": np.mean(confidence_scores),
        "min": np.min(confidence_scores),
        "max": np.max(confidence_scores)
    }
    print(f"\n予測信頼度統計:")
    print(f"  平均: {confidence_stats['mean']:.4f}")
    print(f"  最小: {confidence_stats['min']:.4f}")
    print(f"  最大: {confidence_stats['max']:.4f}")
    
    # 信頼度閾値別の精度
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    threshold_results = {}
    print("\n信頼度閾値別の精度:")
    for threshold in thresholds:
        filtered_indices = [i for i, conf in enumerate(confidence_scores) if conf >= threshold]
        if filtered_indices:
            filtered_true = [true_labels[i] for i in filtered_indices]
            filtered_pred = [pred_labels[i] for i in filtered_indices]
            filtered_acc = accuracy_score(filtered_true, filtered_pred)
            threshold_results[threshold] = {
                "accuracy": filtered_acc,
                "num_samples": len(filtered_indices),
                "total_samples": len(confidence_scores)
            }
            print(f"  閾値 {threshold:.2f}: {filtered_acc:.4f} ({len(filtered_indices)}/{len(confidence_scores)} サンプル)")
    
    # 誤分類サンプルの分析（信頼度順）
    if incorrect_samples:
        print(f"\n誤分類サンプル (信頼度順): {len(incorrect_samples)}/{len(true_labels)}")
        sorted_incorrect = sorted(incorrect_samples, key=lambda x: x["confidence"], reverse=True)
        for i, sample in enumerate(sorted_incorrect[:5]):
            print(f"  サンプル {sample['id']} (真値: {sample['true']}, 予測: {sample['pred']}, 信頼度: {sample['confidence']:.4f}):")
            print(f"    {sample['text']}")
        if len(incorrect_samples) > 5:
            print(f"  ... 他 {len(incorrect_samples)-5} サンプル")
    
    # 結果をファイルに保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 詳細レポートをテキストファイルに保存
        report_filename = os.path.join(output_dir, "clip_evaluation_report.txt")
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("CLIP評価レポート\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"精度: {accuracy:.4f} ({sum(1 for t, p in zip(true_labels, pred_labels) if t == p)}/{len(true_labels)})\n\n")
            f.write("分類レポート:\n")
            f.write(classification_rep + "\n\n")
            f.write("予測信頼度統計:\n")
            f.write(f"  平均: {confidence_stats['mean']:.4f}\n")
            f.write(f"  最小: {confidence_stats['min']:.4f}\n")
            f.write(f"  最大: {confidence_stats['max']:.4f}\n\n")
            f.write("信頼度閾値別の精度:\n")
            for threshold, result in threshold_results.items():
                f.write(f"  閾値 {threshold:.2f}: {result['accuracy']:.4f} ({result['num_samples']}/{result['total_samples']} サンプル)\n")
            
            if incorrect_samples:
                f.write(f"\n誤分類サンプル (信頼度順): {len(incorrect_samples)}/{len(true_labels)}\n")
                sorted_incorrect = sorted(incorrect_samples, key=lambda x: x["confidence"], reverse=True)
                for i, sample in enumerate(sorted_incorrect[:10]):  # 上位10個を保存
                    f.write(f"  サンプル {sample['id']} (真値: {sample['true']}, 予測: {sample['pred']}, 信頼度: {sample['confidence']:.4f}):\n")
                    f.write(f"    {sample['text']}\n")
        
        # JSON形式でも保存
        results_data = {
            "accuracy": float(accuracy),
            "confidence_stats": confidence_stats,
            "threshold_results": threshold_results,
            "incorrect_samples": sorted(incorrect_samples, key=lambda x: x["confidence"], reverse=True)[:20],  # 上位20個
            "total_samples": len(true_labels),
            "misclassified_count": len(incorrect_samples)
        }
        
        json_filename = os.path.join(output_dir, "clip_evaluation_results.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n評価結果を保存しました:")
        print(f"  テキストレポート: {report_filename}")
        print(f"  JSON結果: {json_filename}")
    
    # 混同行列のヒートマップをセーブ（オプション）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(true_labels, pred_labels, labels=categories)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        if output_dir:
            confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_clip.png')
            plt.savefig(confusion_matrix_path)
            print(f"\n混同行列を '{confusion_matrix_path}' に保存しました")
        else:
            plt.savefig('confusion_matrix_clip.png')
            print("\n混同行列を 'confusion_matrix_clip.png' に保存しました")
    except Exception as e:
        print(f"混同行列の保存でエラー: {e}")

    return incorrect_samples, categories

def render_misclassified_samples(incorrect_samples, data_path, output_dir, categories, max_samples=10):
    """誤分類されたサンプルをレンダリングして画像として保存する"""
    print(f"\n誤分類サンプルの点群をレンダリングします（最大{max_samples}個）...")
    
    # 出力ディレクトリの作成
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
        
        plt.suptitle(f"ID: {object_id}, True: {true_label}, Pred: {pred_label}")
        
        # 画像の保存
        filename = f"{output_dir}/misclassified_{i:03d}_id{object_id}_{true_label}_as_{pred_label}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)
        
    print(f"レンダリング完了: {len(samples_to_render)}個のサンプルを{output_dir}に保存しました")

if __name__ == "__main__":
    # ファイルパス
    # result_file = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/MIRU_1/prompt0/ModelNet_classification_modelnet40_data_prompt0.json"
    result_file = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/MIRU_1/prompt1/ModelNet_classification_modelnet40_data_prompt1.json"
    categories_file = "modelnet_config/modelnet40_shape_names_modified.txt"
    # output_dir = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/MIRU_1/prompt0"
    output_dir = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/MIRU_1/prompt1"

    
    # 評価実行
    incorrect_samples, categories = evaluate_with_clip(result_file, categories_file, output_dir)
    
    # Matplotlibによるレンダリング
    data_path = "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
    # render_output_dir = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/MIRU_1/prompt0/misclassified_renders"
    render_output_dir = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_7B_v1.2/MIRU_1/prompt1/misclassified_renders"
    render_misclassified_samples(incorrect_samples, data_path, render_output_dir, categories, max_samples=20)

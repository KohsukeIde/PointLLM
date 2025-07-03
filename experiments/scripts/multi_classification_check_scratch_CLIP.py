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
import glob
import re


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
    target_position = data.get("target_position", None)
    prompt = data.get("prompt", "")
    
    print(f"Found {len(results)} samples in results")
    print(f"Prompt: {prompt}")
    print(f"Target position: {target_position}")
    
    if target_position is None:
        assert False, "target_position is None"
    
    # カテゴリ名の読み込み
    if categories_file and os.path.exists(categories_file):
        categories = [line.rstrip() for line in open(categories_file)]
    else:
        assert False, "categories_file is None"
        # # ファイルがない場合、結果から一意なラベルを収集
        # # 複数オブジェクト形式の場合はobject_detailsから取得
        # unique_categories = set()
        # for sample in results:
        #     if "label_name" in sample:
        #         # 単一オブジェクト形式
        #         unique_categories.add(sample["label_name"])
        #     elif "object_details" in sample:
        #         # 複数オブジェクト形式
        #         for obj_detail in sample.get("object_details", []):
        #             category = obj_detail.get("category")
        #             if category:
        #                 if isinstance(category, list):
        #                     for cat in category:
        #                         unique_categories.add(cat)
        #                 elif isinstance(category, str):
        #                     unique_categories.add(category)
        # categories = sorted(list(unique_categories))
    
    print(f"Using {len(categories)} categories")
    
    # デバッグ情報：実際のデータ形式を確認
    if len(results) > 0:
        sample = results[0]
        print(f"\nサンプルデータ形式確認:")
        print(f"  キー: {sample.keys()}")
        if "object_details" in sample:
            print(f"  object_details[0]: {sample['object_details'][0]}")
        if "model_output" in sample:
            print(f"  model_output: '{sample['model_output'][:100]}...'")
        print()
    
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
        
        # 正解カテゴリとoriginal_dataset_indexを取得
        true_cats = []
        original_indices = []
        for sample in batch:
            if "label_name" in sample:
                # 単一オブジェクト形式
                true_cats.append(sample["label_name"])
                original_indices.append(sample.get("object_id", 0))
            elif "object_details" in sample and len(sample["object_details"]) > 0:
                # 複数オブジェクト形式
                obj_details = sample["object_details"]
                
                if target_position is not None:
                    # 特定のポジションのオブジェクトを正解とする
                    target_obj = None
                    for obj_detail in obj_details:
                        if obj_detail.get("position") == target_position:
                            target_obj = obj_detail
                            break
                    
                    if target_obj:
                        category = target_obj.get("category")
                        if isinstance(category, str):
                            true_cats.append(category)
                        elif isinstance(category, list) and len(category) > 0:
                            true_cats.append(category[0])
                        else:
                            true_cats.append("unknown")
                        original_indices.append(sample.get("object_id", 0))
                    else:
                        # 指定されたポジションが見つからない場合
                        true_cats.append("position_not_found")
                        original_indices.append(sample.get("object_id", 0))
                elif target_position is None:
                    # 比較/判断/カウントタスクまたは古いJSONファイル形式
                    # 古いJSONファイルの場合は最初のオブジェクトを使用
                    obj_detail = obj_details[0]
                    category = obj_detail.get("category")
                    if isinstance(category, str):
                        true_cats.append(category)
                    elif isinstance(category, list) and len(category) > 0:
                        true_cats.append(category[0])
                    else:
                        true_cats.append("unknown")
                    original_indices.append(sample.get("object_id", 0))
                else:
                    # フォールバック
                    obj_detail = obj_details[0]
                    category = obj_detail.get("category")
                    if isinstance(category, str):
                        true_cats.append(category)
                    elif isinstance(category, list) and len(category) > 0:
                        true_cats.append(category[0])
                    else:
                        true_cats.append("unknown")
                    original_indices.append(sample.get("object_id", 0))
            else:
                true_cats.append("unknown")
                original_indices.append(sample.get("object_id", 0))
        
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
        
        for j, (true_cat, pred_idx, conf, output, orig_idx) in enumerate(zip(true_cats, predicted_indices, confidence, model_outputs, original_indices)):
            sample_idx = i + j
            pred_cat = categories[pred_idx]
            
            true_labels.append(true_cat)
            pred_labels.append(pred_cat)
            confidence_scores.append(float(conf))
            
            if true_cat != pred_cat:
                incorrect_samples.append({
                    "id": orig_idx,  # original_dataset_indexを使用
                    "true": true_cat,
                    "pred": pred_cat,
                    "confidence": float(conf),
                    "text": output[:100] + "..." if len(output) > 100 else output
                })
    
    # デバッグ情報：最初の5つのサンプル
    print(f"\n最初の5つのサンプルの詳細:")
    for i in range(min(5, len(true_labels))):
        print(f"  サンプル {i}: 真値='{true_labels[i]}', 予測='{pred_labels[i]}', 信頼度={confidence_scores[i]:.4f}")
        sample_output = results[i]["model_output"][:50] + "..." if len(results[i]["model_output"]) > 50 else results[i]["model_output"]
        print(f"    モデル出力: '{sample_output}'")
    
    # 分類精度の計算
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n精度: {accuracy:.4f} ({sum(1 for t, p in zip(true_labels, pred_labels) if t == p)}/{len(true_labels)})")
    
    if target_position is not None:
        print(f"（ポジション {target_position} のオブジェクトを対象とした評価）")
    else:
        print(f"（最初のオブジェクトまたは比較タスクを対象とした評価）")
    
    # 詳細なレポート - 実際にデータに含まれるカテゴリのみを使用
    unique_categories_in_data = sorted(list(set(true_labels + pred_labels)))
    classification_rep = classification_report(true_labels, pred_labels, labels=unique_categories_in_data, zero_division=0)
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
        
        # ファイル名から プロンプト情報を取得
        filename = os.path.basename(result_file)
        
        # 詳細レポートをテキストファイルに保存
        report_filename = os.path.join(output_dir, f"clip_evaluation_report_{filename.replace('.json', '')}.txt")
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"CLIP評価レポート - {filename}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"prompt: {prompt}\n")
            f.write(f"target_position: {target_position}\n")
            if target_position is not None:
                f.write(f"（ポジション {target_position} のオブジェクトを対象とした評価）\n\n")
            else:
                f.write(f"（最初のオブジェクトまたは比較タスクを対象とした評価）\n\n")
            f.write(f"accuracy: {accuracy:.4f} ({sum(1 for t, p in zip(true_labels, pred_labels) if t == p)}/{len(true_labels)})\n\n")
            f.write("classification_report:\n")
            f.write(classification_rep + "\n\n")
            f.write("confidence_stats:\n")
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
            "prompt": prompt,
            "target_position": target_position,
            "accuracy": float(accuracy),
            "confidence_stats": confidence_stats,
            "threshold_results": threshold_results,
            "incorrect_samples": sorted(incorrect_samples, key=lambda x: x["confidence"], reverse=True)[:20],  # 上位20個
            "total_samples": len(true_labels),
            "misclassified_count": len(incorrect_samples)
        }
        
        json_filename = os.path.join(output_dir, f"clip_evaluation_results_{filename.replace('.json', '')}.json")
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
        
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_categories_in_data)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_categories_in_data, yticklabels=unique_categories_in_data)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {os.path.basename(result_file)}')
        plt.tight_layout()
        if output_dir:
            filename = os.path.basename(result_file)
            confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_clip_{filename.replace(".json", "")}.png')
            plt.savefig(confusion_matrix_path)
            print(f"\n混同行列を '{confusion_matrix_path}' に保存しました")
        else:
            plt.savefig('confusion_matrix_clip.png')
            print("\n混同行列を 'confusion_matrix_clip.png' に保存しました")
    except Exception as e:
        print(f"混同行列の保存でエラー: {e}")

    return incorrect_samples, categories


def render_misclassified_samples(incorrect_samples, data_path, output_dir, categories, max_samples=10, prompt_name=""):
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
        filename = f"{output_dir}/misclassified_{prompt_name}_{i:03d}_id{object_id}_{true_label}_as_{pred_label}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)
        
    print(f"レンダリング完了: {len(samples_to_render)}個のサンプルを{output_dir}に保存しました")


def process_evaluation_multi_directory(evaluation_dir, categories_file, data_path, output_base_dir, render_misclassified=True, max_render_samples=20):
    """
    evaluation_multiディレクトリ内のprompt0からprompt6までのすべてのJSONファイルを処理する
    """
    print(f"Processing files from: {evaluation_dir}")
    
    # JSONファイルを検索（prompt0からprompt6まで）
    json_files = []
    for prompt_num in range(7):  # 0-6
        pattern = os.path.join(evaluation_dir, f"*prompt{prompt_num}*.json")
        files = glob.glob(pattern)
        json_files.extend(files)
    
    json_files.sort()  # ファイル名でソート
    
    if not json_files:
        print("該当するJSONファイルが見つかりません")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for f in json_files:
        print(f"  {os.path.basename(f)}")
    
    # 全体の結果を保存するためのデータ構造
    all_results = {}
    
    # 各ファイルを処理
    for json_file in json_files:
        filename = os.path.basename(json_file)
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        # プロンプト番号を抽出
        match = re.search(r'prompt(\d+)', filename)
        prompt_num = match.group(1) if match else "unknown"
        
        # 出力ディレクトリを作成
        output_dir = os.path.join(output_base_dir, f"prompt{prompt_num}")
        
        try:
            # プロンプト内容を抽出
            with open(json_file, 'r') as f:
                data = json.load(f)
            prompt_content = data.get("prompt", "Unknown prompt")
            results = data.get("results", [])
            
            # CLIP評価を実行
            incorrect_samples, categories = evaluate_with_clip(json_file, categories_file, output_dir)
            
            # レンダリングを実行（オプション）
            if render_misclassified and incorrect_samples:
                render_output_dir = os.path.join(output_dir, "misclassified_renders")
                render_misclassified_samples(
                    incorrect_samples, 
                    data_path, 
                    render_output_dir, 
                    categories, 
                    max_samples=max_render_samples,
                    prompt_name=f"prompt{prompt_num}"
                )
            
            # 結果を全体のサマリーに追加
            total_samples = len(results)
            misclassified_count = len(incorrect_samples)
            accuracy = (total_samples - misclassified_count) / total_samples if total_samples > 0 else 0
            
            all_results[f"prompt{prompt_num}"] = {
                "file": filename,
                "prompt_content": prompt_content,
                "total_samples": total_samples,
                "misclassified_count": misclassified_count,
                "accuracy": accuracy
            }
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # 全体のサマリーレポートを作成
    summary_file = os.path.join(output_base_dir, "evaluation_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CLIP評価サマリーレポート\n")
        f.write("=" * 50 + "\n\n")
        
        for prompt_key, result in sorted(all_results.items()):
            f.write(f"{prompt_key}:\n")
            f.write(f"  ファイル: {result['file']}\n")
            f.write(f"  総サンプル数: {result['total_samples']}\n")
            f.write(f"  誤分類数: {result['misclassified_count']}\n")
            f.write(f"  精度: {result['accuracy']:.4f}\n")
            f.write(f"  プロンプト: {result['prompt_content']}\n\n")
    
    print(f"\n\n全体のサマリーレポートを保存しました: {summary_file}")


if __name__ == "__main__":
    # 設定
    evaluation_dir = "/groups/gag51404/ide/PointLLM/outputs/PointLLM_train_stage2/PointLLM_train_stage2_naive_batch/evaluation_multi"
    categories_file = "/groups/gag51404/ide/PointLLM/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt"
    data_path = "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
    output_base_dir = "/groups/gag51404/ide/PointLLM/evaluation/PointLLM_train_stage2/evaluation_multi_clip_results_naive_batch"
    
    # すべてのprompt0-6のファイルを処理
    process_evaluation_multi_directory(
        evaluation_dir=evaluation_dir,
        categories_file=categories_file,
        data_path=data_path,
        output_base_dir=output_base_dir,
        render_misclassified=True,  # 誤分類サンプルをレンダリングするかどうか
        max_render_samples=5      # レンダリングする最大サンプル数
    )
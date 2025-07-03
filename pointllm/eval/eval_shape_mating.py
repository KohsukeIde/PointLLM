# pointllm/eval/eval_shape_mating.py
"""
Shape Mating Pair Discovery Evaluation Script with GPT-4 Evaluator

This script evaluates a model's ability to identify which two parts from three given 3D parts 
form a mating pair. It uses GPT-4 as an evaluator to interpret model responses semantically,
avoiding the brittleness of keyword matching.

Usage:
    python pointllm/eval/eval_shape_mating.py \
        --model_name ./outputs/PointLLM_train_stage3/PointLLM_train_stage3_shape_mating_batch \
        --anno_path ./data/anno_data/dataset_generator/shape_mating_pair_discovery_test_0K_gpt.json \
        --output_dir ./evaluation_results/
"""

import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import re
import json
import time
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

# GPT Evaluator imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI module not available. GPT evaluation will be disabled.")

class GPTEvaluator:
    """GPT-4を使用してモデル応答を評価するクラス"""
    
    def __init__(self, api_key=None, model_name="gpt-4o", max_retries=3):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI module is required for GPT evaluation")
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key_from_env = os.getenv("OPENAI_API_KEY")
            if not api_key_from_env:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key_from_env)
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
    
    def get_evaluation_prompt(self, model_output):
        """GPT-4への評価指示プロンプトを生成"""
        prompt = f"""あなたは、AIモデルの性能を評価する専門家です。
これから、別のAIモデルが生成した応答テキストを提示します。
このAIは、「提示された3つの3D部品の中から、嵌合するペアはどれか？」という質問に答えました。

応答テキストを注意深く読み、AIが「何番目」と「何番目」の部品がペアだと結論付けたかを解釈してください。

# ルール
- 位置は0から始まるインデックス（first=0, second=1, third=2）で回答してください。
- 必ず2つの異なるインデックスを特定してください。
- 応答テキストから2つのインデックスを特定できない場合は、空のリスト `[]` を返してください。
- 「最初」「1番目」「first」などは0、「2番目」「second」などは1、「3番目」「third」などは2として扱ってください。

# 出力形式
必ず以下のJSON形式で回答してください。他の説明や文章は一切含めないでください。
{{"predicted_indices": [インデックス1, インデックス2]}}

# 応答テキスト
{model_output}

# JSON出力:"""
        return prompt
    
    def evaluate_response(self, model_output):
        """
        モデルの応答をGPT-4で評価し、予測されたインデックスを返す
        
        Args:
            model_output (str): 評価対象のモデル応答
            
        Returns:
            list: 予測されたインデックスのリスト（0-indexed）
        """
        self.call_count += 1
        
        prompt = self.get_evaluation_prompt(model_output)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1,  # より一貫した結果のために低い温度
                    timeout=30
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # JSONパース（マークダウンコードブロックの除去）
                try:
                    # マークダウンコードブロックを除去
                    cleaned_text = result_text.strip()
                    if cleaned_text.startswith('```json'):
                        cleaned_text = cleaned_text[7:]  # ```json を除去
                    if cleaned_text.startswith('```'):
                        cleaned_text = cleaned_text[3:]   # ``` を除去
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3]  # 末尾の ``` を除去
                    cleaned_text = cleaned_text.strip()
                    
                    result_json = json.loads(cleaned_text)
                    predicted_indices = result_json.get("predicted_indices", [])
                    
                    # バリデーション
                    if isinstance(predicted_indices, list):
                        # インデックスの有効性チェック
                        valid_indices = []
                        for idx in predicted_indices:
                            if isinstance(idx, int) and 0 <= idx <= 2:
                                valid_indices.append(idx)
                        
                        # 重複を除いて昇順にソート
                        final_indices = sorted(list(set(valid_indices)))
                        
                        self.success_count += 1
                        return final_indices
                    
                except json.JSONDecodeError:
                    print(f"Warning: GPT response is not valid JSON: {cleaned_text[:100]}...")
                
            except Exception as e:
                print(f"GPT API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # 全ての試行が失敗した場合
        self.error_count += 1
        print(f"Warning: GPT evaluation failed for output: {model_output[:100]}...")
        return []  # 空のリストを返してフォールバック
    
    def get_stats(self):
        """評価統計を取得"""
        return {
            "total_calls": self.call_count,
            "successful_calls": self.success_count,
            "failed_calls": self.error_count,
            "success_rate": self.success_count / max(1, self.call_count)
        }

def parse_response_to_indices_fallback(text):
    """
    フォールバック用のキーワードマッチング（GPTが失敗した場合のみ使用）
    
    Args:
        text (str): モデルの応答テキスト
        
    Returns:
        list: 抽出されたインデックスのリスト（0-indexed）
    """
    text = text.lower()
    indices = []
    
    # 序数から数字へのマッピング
    ordinal_map = {
        "first": 0, "second": 1, "third": 2,
        "1st": 0, "2nd": 1, "3rd": 2,
        "one": 0, "two": 1, "three": 2,
        "①": 0, "②": 1, "③": 2,
        "1": 0, "2": 1, "3": 2
    }
    
    # 各序数/数字を検索してインデックスを抽出
    for word, index in ordinal_map.items():
        if word in text:
            indices.append(index)
    
    # 数字のパターンも検索（例: "1と3", "parts 1 and 3"など）
    number_pattern = re.findall(r'\b([123])\b', text)
    for num_str in number_pattern:
        index = int(num_str) - 1  # 1-indexed to 0-indexed
        if 0 <= index <= 2:
            indices.append(index)
    
    # 重複を除いて昇順にソート
    return sorted(list(set(indices)))

def init_model(args):
    """モデルとトークナイザを初期化する"""
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:0")
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        use_cache=True, 
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_shape_mating_dataset(anno_path, data_root):
    """
    Shape Mating データセットを読み込み
    
    Args:
        anno_path (str): データセットのJSONファイルパス
        data_root (str): 点群データのルートディレクトリ
        
    Returns:
        list: データセットサンプルのリスト
    """
    print(f"Loading Shape Mating dataset from {anno_path}")
    
    with open(anno_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    # 点群ファイルの存在確認
    valid_samples = []
    for sample in dataset:
        object_ids = sample['object_ids']
        valid = True
        
        # 各点群ファイルの存在確認
        for obj_path in object_ids:
            if not os.path.isabs(obj_path):
                # 相対パスの場合、data_rootと結合
                full_path = os.path.join(data_root, obj_path)
            else:
                full_path = obj_path
                
            if not os.path.exists(full_path):
                print(f"Warning: Point cloud file not found: {full_path}")
                valid = False
                break
        
        if valid:
            valid_samples.append(sample)
    
    print(f"Found {len(valid_samples)} valid samples with existing point cloud files")
    return valid_samples

def get_shape_mating_dataloader(samples, data_root, batch_size, use_color, shuffle=False, num_workers=4):
    """Shape Mating用のDataLoaderを作成"""
    
    class ShapeMatingDataset(torch.utils.data.Dataset):
        def __init__(self, samples, data_root, use_color):
            self.samples = samples
            self.data_root = data_root
            self.use_color = use_color
        
        def __len__(self):
            return len(self.samples)
        
        def _load_point_cloud(self, obj_path):
            """点群データを読み込み"""
            if not os.path.isabs(obj_path):
                full_path = os.path.join(self.data_root, obj_path)
            else:
                full_path = obj_path
            
            # .npyファイルを直接読み込み
            if full_path.endswith('.npy'):
                point_cloud = np.load(full_path)
                point_cloud = torch.from_numpy(point_cloud).float()
                
                # Shape Matingデータは3チャンネル(XYZ)なので、ダミーRGBを追加
                if point_cloud.shape[1] == 3 and self.use_color:
                    dummy_rgb = torch.ones(point_cloud.shape[0], 3) * 0.5
                    point_cloud = torch.cat([point_cloud, dummy_rgb], dim=1)
                elif point_cloud.shape[1] == 6 and not self.use_color:
                    point_cloud = point_cloud[:, :3]
                    
                return point_cloud
            else:
                raise ValueError(f"Unsupported file format: {full_path}")
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            object_ids = sample['object_ids']
            
            # 3つの点群を読み込み
            point_clouds_list = []
            for obj_path in object_ids:
                pc = self._load_point_cloud(obj_path)
                point_clouds_list.append(pc)
            
            # 複数点群をテンソルに結合 (3, N, C)
            multi_point_clouds = torch.stack(point_clouds_list, dim=0)
            
            return {
                'point_clouds': multi_point_clouds,
                'conversations': sample['conversations'],
                'metadata': sample['metadata'],
                'object_ids': object_ids,
                'sample_idx': idx
            }
    
    def collate_fn(batch):
        """バッチ用のcollate関数"""
        # point_cloudsをスタック
        point_clouds = torch.stack([item['point_clouds'] for item in batch])
        
        # その他のフィールドをリストとして収集
        batched_data = {
            'point_clouds': point_clouds,
            'conversations': [item['conversations'] for item in batch],
            'metadata': [item['metadata'] for item in batch],
            'object_ids': [item['object_ids'] for item in batch],
            'sample_idx': [item['sample_idx'] for item in batch]
        }
        return batched_data
    
    dataset = ShapeMatingDataset(samples, data_root, use_color)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, 
                    do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    """推論関数"""
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    # キャッシュクリア
    torch.cuda.empty_cache()

    return outputs

def start_shape_mating_evaluation(model, tokenizer, conv, dataloader, output_dir, output_file, 
                                 use_gpt_eval=True, gpt_evaluator=None):
    """Shape Mating評価の実行（GPT-4評価対応版）"""
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    # 点群トークンの設定
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    replace_token = point_backbone_config['default_point_patch_token'] * point_token_len
    if point_backbone_config['mm_use_point_start_end']:
        replace_token = point_backbone_config['default_point_start_token'] + replace_token + point_backbone_config['default_point_end_token']

    results = {"task_type": "shape_mating_pair_discovery", "results": []}
    
    # 統計変数の初期化
    total_samples = 0
    correct_predictions = 0
    task_stats = {1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0}, 3: {"correct": 0, "total": 0}}
    gpt_fallback_count = 0

    print(f"[INFO] Using {'GPT-4 evaluation' if use_gpt_eval and gpt_evaluator else 'keyword matching fallback'}")

    for batch in tqdm(dataloader, desc="Evaluating Shape Mating"):
        point_clouds = batch['point_clouds'].cuda().to(model.dtype)
        batch_size = point_clouds.shape[0]
        
        # バッチ内の各サンプルを処理
        input_ids_batch = []
        for i in range(batch_size):
            conversations = batch['conversations'][i]
            
            # human側のメッセージ（プロンプト）を取得
            human_message = conversations[0]['value']
            
            # <pc_数字>パターンを点群トークンに置換
            processed_prompt = human_message
            pc_placeholders = re.findall(r'<pc_\d+>', processed_prompt)
            for placeholder in pc_placeholders:
                processed_prompt = processed_prompt.replace(placeholder, replace_token, 1)
            
            # 会話テンプレートの準備
            conv.messages = []
            conv.append_message(conv.roles[0], processed_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            inputs = tokenizer([prompt], return_tensors="pt")
            input_ids_batch.append(inputs.input_ids.squeeze(0))
        
        # パディングしてバッチ化
        max_len = max(ids.shape[0] for ids in input_ids_batch)
        padded_input_ids = []
        for ids in input_ids_batch:
            if ids.shape[0] < max_len:
                padding = torch.full((max_len - ids.shape[0],), tokenizer.pad_token_id, dtype=ids.dtype)
                padded_ids = torch.cat([ids, padding])
            else:
                padded_ids = ids
            padded_input_ids.append(padded_ids)
        
        input_ids = torch.stack(padded_input_ids).cuda()
        
        stopping_criteria = KeywordsStoppingCriteria([stop_str, tokenizer.eos_token], tokenizer, input_ids)
        
        # 推論実行
        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria)
        
        # 結果の処理
        for i in range(batch_size):
            model_output_text = outputs[i]
            metadata = batch['metadata'][i]
            conversations = batch['conversations'][i]
            
            # ★★★ GPT-4評価または従来の方法で予測インデックスを抽出 ★★★
            if use_gpt_eval and gpt_evaluator:
                predicted_indices = gpt_evaluator.evaluate_response(model_output_text)
                evaluation_method = "gpt"
                
                # GPTが失敗した場合のフォールバック
                if not predicted_indices:
                    predicted_indices = parse_response_to_indices_fallback(model_output_text)
                    evaluation_method = "fallback"
                    gpt_fallback_count += 1
            else:
                predicted_indices = parse_response_to_indices_fallback(model_output_text)
                evaluation_method = "keyword"
            
            # 正解インデックスを取得
            ground_truth_indices = metadata["correct_pair_indices"]
            
            # 正誤判定
            is_correct = (sorted(predicted_indices) == sorted(ground_truth_indices))
            
            # 統計更新
            total_samples += 1
            if is_correct:
                correct_predictions += 1
            
            task_type = metadata["task_type"]
            task_stats[task_type]["total"] += 1
            if is_correct:
                task_stats[task_type]["correct"] += 1
            
            # 結果を保存
            result_entry = {
                "sample_idx": batch['sample_idx'][i],
                "object_ids": batch["object_ids"][i],
                "prompt": conversations[0]["value"],
                "ground_truth_response": conversations[1]["value"],
                "model_output": model_output_text,
                "ground_truth_indices": ground_truth_indices,
                "predicted_indices": predicted_indices,
                "is_correct": is_correct,
                "task_type": task_type,
                "task_difficulty": metadata.get("difficulty", "unknown"),
                "evaluation_method": evaluation_method
            }
            results["results"].append(result_entry)
    
    # 最終統計の計算
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # タスク別精度の計算
    task_accuracies = {}
    for task_type, stats in task_stats.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            task_accuracies[f"task_{task_type}"] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"]
            }
    
    # GPT評価統計
    gpt_stats = {}
    if use_gpt_eval and gpt_evaluator:
        gpt_stats = gpt_evaluator.get_stats()
        gpt_stats["fallback_count"] = gpt_fallback_count
    
    # 統計情報を結果に追加
    results["statistics"] = {
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "overall_accuracy": overall_accuracy,
        "task_accuracies": task_accuracies,
        "evaluation_method": "gpt" if use_gpt_eval else "keyword",
        "gpt_evaluation_stats": gpt_stats
    }
    
    # 結果をファイルに保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 統計を表示
    print(f"\n{'='*60}")
    print(f"Shape Mating Pair Discovery Evaluation Results")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    print(f"\nTask-specific accuracies:")
    for task_type, accuracy_info in task_accuracies.items():
        task_num = task_type.split('_')[1]
        difficulty_map = {"1": "Easy", "2": "Medium", "3": "Hard"}
        difficulty = difficulty_map.get(task_num, "Unknown")
        print(f"  Task {task_num} ({difficulty}): {accuracy_info['accuracy']:.4f} "
              f"({accuracy_info['correct']}/{accuracy_info['total']})")
    
    # GPT評価統計の表示
    if use_gpt_eval and gpt_evaluator:
        print(f"\nGPT-4 Evaluation Statistics:")
        print(f"  Total GPT calls: {gpt_stats['total_calls']}")
        print(f"  Successful calls: {gpt_stats['successful_calls']}")
        print(f"  Failed calls: {gpt_stats['failed_calls']}")
        print(f"  Success rate: {gpt_stats['success_rate']:.4f}")
        print(f"  Fallback used: {gpt_fallback_count} times")
    
    print(f"\nResults saved to: {output_path}")
    
    return results

def main(args):
    # 乱数シードを固定
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 出力ディレクトリとファイル名の設定
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_name), "shape_mating_evaluation")
    
    if args.output_file is None:
        model_basename = os.path.basename(os.path.expanduser(args.model_name))
        dataset_basename = os.path.basename(args.anno_path).replace('.json', '')
        eval_method = "gpt" if args.use_gpt_eval else "keyword"
        args.output_file = f"shape_mating_evaluation_{model_basename}_{dataset_basename}_{eval_method}.json"
    
    output_file_path = os.path.join(args.output_dir, args.output_file)
    
    print(f"Shape Mating Pair Discovery Evaluation")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.anno_path}")
    print(f"Evaluation method: {'GPT-4' if args.use_gpt_eval else 'Keyword matching'}")
    print(f"Output: {output_file_path}")
    
    # 既存の結果ファイルが存在するかチェック
    if os.path.exists(output_file_path) and not args.force_regenerate:
        print(f"[INFO] Results file already exists: {output_file_path}")
        print(f"[INFO] Use --force_regenerate to overwrite")
        
        # 既存結果を読み込んで統計を表示
        with open(output_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if "statistics" in results:
            stats = results["statistics"]
            print(f"\nExisting results summary:")
            print(f"Overall accuracy: {stats['overall_accuracy']:.4f}")
            for task_type, accuracy_info in stats['task_accuracies'].items():
                print(f"  {task_type}: {accuracy_info['accuracy']:.4f}")
        return
    
    # GPT評価器の初期化
    gpt_evaluator = None
    if args.use_gpt_eval:
        if not OPENAI_AVAILABLE:
            print("Error: OpenAI module not available. Install with: pip install openai")
            print("Falling back to keyword matching evaluation.")
            args.use_gpt_eval = False
        else:
            try:
                gpt_evaluator = GPTEvaluator(
                    api_key=args.gpt_api_key,
                    model_name=args.gpt_model,
                    max_retries=3
                )
                print(f"[INFO] GPT-4 evaluator initialized with model: {args.gpt_model}")
            except Exception as e:
                print(f"Error initializing GPT evaluator: {e}")
                print("Falling back to keyword matching evaluation.")
                args.use_gpt_eval = False
    
    # データセット読み込み
    samples = load_shape_mating_dataset(args.anno_path, args.data_root)
    
    if len(samples) == 0:
        print("Error: No valid samples found. Please check data paths.")
        return
    
    # DataLoader作成
    dataloader = get_shape_mating_dataloader(
        samples, args.data_root, args.batch_size, args.use_color, 
        args.shuffle, args.num_workers
    )
    
    # モデル初期化
    model, tokenizer, conv = init_model(args)
    
    # 評価実行
    print(f'[INFO] Starting Shape Mating evaluation...')
    results = start_shape_mating_evaluation(
        model, tokenizer, conv, dataloader, args.output_dir, args.output_file,
        use_gpt_eval=args.use_gpt_eval, gpt_evaluator=gpt_evaluator
    )
    
    # リソース解放
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print(f"[INFO] Evaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape Mating Pair Discovery Evaluation with GPT-4 Evaluator")
    
    # モデル関連
    parser.add_argument("--model_name", type=str, 
                        default="/groups/gag51404/ide/PointLLM/outputs/PointLLM_train_stage3/PointLLM_train_stage3_shape_mating_batch",
                        help="Path to the model checkpoint")
    
    # データ関連
    parser.add_argument("--anno_path", type=str, 
                        default="/groups/gag51404/ide/PointLLM/data/anno_data/dataset_generator/shape_mating_pair_discovery_test_0K_gpt.json",
                        help="Path to the shape mating annotation JSON file")
    parser.add_argument("--data_root", type=str, 
                        default="/groups/gag51402/datasets/ORB3D/objaverse_PWN_filtered",
                        help="Root directory containing point cloud data")
    parser.add_argument("--use_color", action="store_true", default=True,
                        help="Use color information")
    
    # GPT評価関連
    parser.add_argument("--use_gpt_eval", action="store_true", default=True,
                        help="Use GPT-4 for evaluation instead of keyword matching")
    parser.add_argument("--gpt_api_key", type=str, default=None,
                        help="OpenAI API key (optional, can use OPENAI_API_KEY env var)")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o",
                        help="GPT model to use for evaluation")
    
    # 実行設定
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--shuffle", type=bool, default=False,
                        help="Shuffle data (should be False for reproducible results)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    
    # 出力設定
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for evaluation results")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file name")
    parser.add_argument("--force_regenerate", action="store_true",
                        help="Force regenerate even if results file exists")
    
    args = parser.parse_args()
    main(args) 
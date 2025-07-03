#!/usr/bin/env python3

import os
import json
import random
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import re
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ComplexStage2DatasetGenerator:
    """Complex instruction用のStage2複数点群データセット生成器（GPT API使用版）"""
    
    def __init__(self, data_dir: str, output_dir: str, seed: int = 42, use_gpt_api: bool = False, 
                 api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo", 
                 save_comparison: bool = False, max_workers: int = 10):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        self.use_gpt_api = use_gpt_api
        self.model_name = model_name
        self.save_comparison = save_comparison
        self.max_workers = max_workers
        
        random.seed(seed)
        
        # OpenAI API設定
        if self.use_gpt_api:
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                api_key_from_env = os.getenv("OPENAI_API_KEY")
                if not api_key_from_env:
                    raise ValueError("OpenAI API key is required when use_gpt_api=True. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
                self.openai_client = OpenAI(api_key=api_key_from_env)
        else:
            self.openai_client = None
        
        # 入力データファイルのパス
        self.complex_instruction_file = os.path.join(data_dir, "PointLLM_complex_instruction_70K.json")
        self.brief_description_file = os.path.join(data_dir, "PointLLM_brief_description_660K_filtered.json")
        
        # データ読み込み
        self.complex_data = self._load_json_data(self.complex_instruction_file)
        self.brief_data = self._load_json_data(self.brief_description_file)
        
        print(f"Loaded {len(self.complex_data)} complex instruction samples")
        print(f"Loaded {len(self.brief_data)} brief description samples")
        print(f"GPT API usage: {'Enabled' if self.use_gpt_api else 'Disabled'}")
        if self.use_gpt_api:
            print(f"Model: {self.model_name}")
        
        # 1サンプルあたりに含める点群の数の選択肢
        self.num_pc_choices = [2, 3]
        
        # フォールバック用の基本順序表現テンプレート（"first"を含む）
        self.fallback_position_templates = {
            "first": ["first", "1st", "initial"],
            "second": ["second", "2nd", "middle"],
            "third": ["third", "3rd", "last"]
        }
        
        # API呼び出しの統計（スレッドセーフ）
        self.api_call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0
        }
        self.stats_lock = threading.Lock()
        
    def _load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """JSONデータを読み込む"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _call_gpt_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """GPT APIを呼び出す（リトライ機能付き、スレッドセーフ）"""
        for attempt in range(max_retries):
            try:
                with self.stats_lock:
                    self.api_call_stats["total_calls"] += 1
                
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3,
                    timeout=30
                )
                
                result = response.choices[0].message.content.strip()
                with self.stats_lock:
                    self.api_call_stats["successful_calls"] += 1
                return result
                
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    with self.stats_lock:
                        self.api_call_stats["failed_calls"] += 1
                    return None
    
    def _generate_position_template_randomly(self, position_key: str) -> str:
        """位置テンプレートをランダムに選択（コスト効率を重視）"""
        available_templates = self.fallback_position_templates[position_key]
        return random.choice(available_templates)
    
    def _replace_demonstratives_with_gpt(self, text: str, target_position_template: str) -> str:
        """GPTを使って指示代名詞や代名詞を置き換える"""
        if not self.use_gpt_api:
            return self._replace_demonstratives_fallback(text, target_position_template)
        
        prompt = f"""Rewrite the following instruction by adding position information while preserving the original context and object type.

Original instruction: "{text}"

Rules:
1. Replace "this", "that", "these", "those" with "the {target_position_template}"
2. Replace "it" with "the {target_position_template} object" or "the {target_position_template} item"
3. Replace phrases like "given object", "presented item" with "the {target_position_template} object"
4. When referring to specific object types, PRESERVE the object type and add position information:
   - "the book" → "the {target_position_template} book" or "the book in the {target_position_template} point cloud"
   - "this airplane" → "the {target_position_template} airplane" 
   - "the car" → "the {target_position_template} car"
5. Keep the original context and meaning intact - only add position clarification
6. Prefer natural expressions like "the {target_position_template} [object]" over "the [object] in the {target_position_template} point cloud"
7. Keep the rest of the text exactly the same
8. Return only the rewritten instruction, no explanation
9. Negative Examples (Do NOT change these):
   - "How does it compare to a real car?" → "How does the {target_position_template} object compare to a real car?" (Here, "a real car" should NOT be changed)
   - "Is this similar to other books?" → "Is the {target_position_template} book similar to other books?" (Here, "other books" should NOT be changed)
   - "What makes it different from typical planes?" → "What makes the {target_position_template} plane different from typical planes?" (Here, "typical planes" should NOT be changed)

Rewritten instruction:"""
        
        result = self._call_gpt_api(prompt)
        
        if result:
            return result
        else:
            # フォールバックを使用
            with self.stats_lock:
                self.api_call_stats["fallback_used"] += 1
            return self._replace_demonstratives_fallback(text, target_position_template)
    
    def _replace_demonstratives_fallback(self, text: str, target_position_template: str) -> str:
        """フォールバック用の正規表現ベースの置換"""
        patterns = [
            # 基本的な指示代名詞
            (r'\bthis\b(?!\s+\w)', f'the {target_position_template}'),
            (r'\bthat\b(?!\s+\w)', f'the {target_position_template}'),
            (r'\bthese\b', f'the {target_position_template}'),
            (r'\bthose\b', f'the {target_position_template}'),
            
            # 指示代名詞 + 名詞 → 位置指定に変更
            (r'\bthis\s+([a-zA-Z]+)\b', rf'the {target_position_template} \1'),
            (r'\bthat\s+([a-zA-Z]+)\b', rf'the {target_position_template} \1'),
            
            # 定冠詞 + 名詞 → 位置指定 + 同じ名詞（オブジェクトの種類は保持）
            (r'\bthe\s+(book|car|chair|table|airplane|plane|bottle|cup|phone|laptop|computer|toy|doll|ball|box|bag|hat|shoe|glass|watch|camera|bike|bicycle|truck|bus|train|boat|ship|house|building|tree|flower|cat|dog|bird|animal)\b', 
             rf'the {target_position_template} \1'),
            
            # 代名詞
            (r'\bit\b', f'the {target_position_template} object'),
            
            # その他の表現（オブジェクト名は保持）
            (r'\bgiven\s+([a-zA-Z]+)\b', rf'the {target_position_template} \1'),
            (r'\bpresented\s+([a-zA-Z]+)\b', rf'the {target_position_template} \1'),
            (r'\bshown\s+([a-zA-Z]+)\b', rf'the {target_position_template} \1'),
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _process_batch(self, batch_data: List[tuple]) -> List[Dict[str, Any]]:
        """バッチデータを並列処理"""
        batch_results = []
        for target_sample, background_samples, target_index in batch_data:
            try:
                result = self._create_complex_stage2_sample(target_sample, background_samples, target_index)
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing sample {target_sample.get('object_id', 'unknown')}: {e}")
                continue
        return batch_results
    
    def _create_complex_stage2_sample(self, target_sample: Dict[str, Any], 
                                    background_samples: List[Dict[str, Any]], 
                                    target_index: int) -> Dict[str, Any]:
        """Complex instruction用のStage2サンプルを作成（conversation_typeに応じて処理）"""
        num_pcs = len(background_samples) + 1  # target + background
        
        # 全てのサンプルを配列に配置（ターゲットをランダム位置に挿入）
        all_samples = background_samples.copy()
        all_samples.insert(target_index, target_sample)
        
        # オブジェクトIDを取得
        object_ids = []
        for sample in all_samples:
            if isinstance(sample.get("object_id"), list):
                object_ids.append(sample["object_id"][0])
            else:
                object_ids.append(sample.get("object_id", "unknown"))
        
        # 順序語のキーを取得
        position_keys = ["first", "second", "third"]
        position_key = position_keys[target_index]
        
        # プレースホルダーの作成（naiveと同じ形式）
        identifier_prefix = ' '.join(['<point>'] * num_pcs)
        
        # ターゲットサンプルの会話を取得
        original_conversations = target_sample.get("conversations", [])
        conversation_type = target_sample.get("conversation_type", "unknown")
        
        if not original_conversations:
            raise ValueError(f"No conversations found in target sample {target_sample.get('object_id', 'unknown')}")
        
        # 会話タイプに応じて処理
        if conversation_type == "multi_round":
            # multi_roundの場合：最初のhuman発話のみを修正し、残りはそのまま保持
            new_conversations = []
            first_human_processed = False
            
            for conv in original_conversations:
                if conv.get("from") == "human" and not first_human_processed:
                    # 最初のhuman発話を修正
                    original_value = conv.get("value", "")
                    cleaned_value = original_value.replace("<point>\n", "").replace("<point>", "")
                    
                    # 順序語テンプレートをランダムに取得
                    target_position_template = self._generate_position_template_randomly(position_key)
                    
                    # 指示代名詞や代名詞を置換（GPTまたはフォールバック）
                    modified_value = self._replace_demonstratives_with_gpt(cleaned_value, target_position_template)
                    
                    # 新しいプレースホルダーと修正した指示文を結合
                    new_value = f"{identifier_prefix}\n{modified_value}"
                    
                    new_conversations.append({
                        "from": "human",
                        "value": new_value
                    })
                    first_human_processed = True
                else:
                    # 2回目以降のhuman発話やgpt発話はそのまま保持
                    new_conversations.append(conv)
        
        else:
            # single_roundやdetailed_descriptionの場合：最初の一問一答のみ使用
            first_human_conv = None
            first_gpt_conv = None
            
            for i, conv in enumerate(original_conversations):
                if conv.get("from") == "human" and first_human_conv is None:
                    first_human_conv = conv
                    # 次のGPT応答を探す
                    for j in range(i + 1, len(original_conversations)):
                        if original_conversations[j].get("from") == "gpt":
                            first_gpt_conv = original_conversations[j]
                            break
                    break
            
            if not first_human_conv or not first_gpt_conv:
                raise ValueError(f"No valid human-gpt conversation pair found in target sample {target_sample.get('object_id', 'unknown')}")
            
            # 人間側の発話を書き換え
            original_value = first_human_conv.get("value", "")
            cleaned_value = original_value.replace("<point>\n", "").replace("<point>", "")
            
            # 順序語テンプレートをランダムに取得
            target_position_template = self._generate_position_template_randomly(position_key)
            
            # 指示代名詞や代名詞を置換（GPTまたはフォールバック）
            modified_value = self._replace_demonstratives_with_gpt(cleaned_value, target_position_template)
            
            # 新しいプレースホルダーと修正した指示文を結合
            new_value = f"{identifier_prefix}\n{modified_value}"
            
            # 新しい会話データを作成（一問一答のみ）
            new_conversations = [
                {
                    "from": "human",
                    "value": new_value
                },
                {
                    "from": "gpt",
                    "value": first_gpt_conv.get("value", "")
                }
            ]
        
        # 新しいサンプルを構築
        new_sample = {
            "object_ids": object_ids,
            "conversations": new_conversations,
            "metadata": {
                "num_point_clouds": num_pcs,
                "target_index": target_index,
                "target_position": position_key,
                "target_position_template": target_position_template if 'target_position_template' in locals() else "unknown",
                "source_object_id": target_sample.get("object_id", "unknown"),
                "conversation_type": conversation_type,
                "background_object_ids": [sample.get("object_id", "unknown") for sample in background_samples],
                "generation_type": f"complex_instruction_stage2_{conversation_type}",
                "gpt_api_used": self.use_gpt_api,
                "original_conversation_length": len(original_conversations)
            }
        }
        
        # 比較用データも作成（オプション有効時のみ）
        if self.save_comparison:
            comparison_sample = {
                "sample_id": len(getattr(self, '_comparison_data', [])),  # 一意ID
                "object_ids": object_ids,
                "target_index": target_index,
                "target_position": position_key,
                "target_position_template": target_position_template if 'target_position_template' in locals() else "unknown",
                "source_object_id": target_sample.get("object_id", "unknown"),
                "conversation_type": conversation_type,
                "original_conversations": original_conversations,
                "modified_conversations": new_conversations,
                "background_object_ids": [sample.get("object_id", "unknown") for sample in background_samples],
                "gpt_api_used": self.use_gpt_api
            }
            
            # 比較データを保存（初回作成時）
            if not hasattr(self, '_comparison_data'):
                self._comparison_data = []
            self._comparison_data.append(comparison_sample)
        
        return new_sample
    
    def generate_complex_stage2_dataset(self, num_samples: int = 70000) -> List[Dict[str, Any]]:
        """Complex instruction用のStage2データセットを生成（並列化対応）"""
        if len(self.complex_data) < 1:
            raise ValueError("Need at least 1 sample in complex instruction data")
        if len(self.brief_data) < 2:
            raise ValueError("Need at least 2 samples in brief description data for background")
        
        print(f"Generating {num_samples} Complex Stage2 samples...")
        print(f"Complex instruction data size: {len(self.complex_data)}")
        print(f"Brief description data size: {len(self.brief_data)}")
        print(f"Using {self.max_workers} workers for parallel processing")
        
        # データ生成パラメータを事前に準備
        sample_params = []
        for i in range(num_samples):
            # 1. 1サンプルあたりの点群数をランダムに決定 (2 or 3)
            num_pcs = random.choice(self.num_pc_choices)
            
            # 2. complex_instructionから1つのサンプルを「主役」として選択
            target_sample = random.choice(self.complex_data)
            
            # 3. brief_descriptionから背景用のサンプルを選択（num_pcs - 1個）
            background_samples = random.sample(self.brief_data, num_pcs - 1)
            
            # 4. ターゲットの位置をランダムに決定 (0-indexed)
            target_index = random.randint(0, num_pcs - 1)
            
            sample_params.append((target_sample, background_samples, target_index))
        
        # バッチサイズを計算（ワーカー数の2倍程度が効率的）
        batch_size = max(1, len(sample_params) // (self.max_workers * 2))
        batches = [sample_params[i:i + batch_size] for i in range(0, len(sample_params), batch_size)]
        
        print(f"Processing {len(batches)} batches with batch size {batch_size}")
        
        new_dataset = []
        
        # 並列処理実行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # バッチを並列実行
            future_to_batch = {executor.submit(self._process_batch, batch): batch for batch in batches}
            
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        new_dataset.extend(batch_results)
                        pbar.update(1)
                        
                        # 進捗レポート（GPT APIを使用している場合）
                        if self.use_gpt_api and len(new_dataset) % 1000 == 0:
                            with self.stats_lock:
                                stats = self.api_call_stats.copy()
                            print(f"\nAPI Stats (samples: {len(new_dataset)}): "
                                  f"Total={stats['total_calls']}, "
                                  f"Success={stats['successful_calls']}, "
                                  f"Failed={stats['failed_calls']}, "
                                  f"Fallback={stats['fallback_used']}")
                        
                    except Exception as e:
                        print(f"Batch processing failed: {e}")
                        continue
        
        print(f"Generated {len(new_dataset)} Complex Stage2 samples")
        
        if self.use_gpt_api:
            print("\nFinal API Statistics:")
            print(f"  Total API calls: {self.api_call_stats['total_calls']}")
            print(f"  Successful calls: {self.api_call_stats['successful_calls']}")
            print(f"  Failed calls: {self.api_call_stats['failed_calls']}")
            print(f"  Fallback used: {self.api_call_stats['fallback_used']}")
            success_rate = (self.api_call_stats['successful_calls'] / max(1, self.api_call_stats['total_calls'])) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        return new_dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], 
                    filename: str = "complex_instruction_stage2_multi_pc_70K.json"):
        """データセットを保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"Saving complex stage2 dataset to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # 比較データを保存（オプション有効時のみ）
        if self.save_comparison and hasattr(self, '_comparison_data') and self._comparison_data:
            comparison_filename = filename.replace('.json', '_comparison.json')
            comparison_path = os.path.join(self.output_dir, comparison_filename)
            print(f"Saving comparison data to {comparison_path}...")
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(self._comparison_data, f, indent=2, ensure_ascii=False)
            print(f"Comparison data saved to {comparison_path}")
        
        # 統計情報を生成・保存
        stats = self._generate_statistics(dataset)
        stats_filename = filename.replace('.json', '_stats.json')
        stats_path = os.path.join(self.output_dir, stats_filename)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Complex stage2 dataset saved to {output_path}")
        print(f"Statistics saved to {stats_path}")
        
        return output_path
    
    def _generate_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """データセット統計を生成"""
        total_samples = len(dataset)
        
        # 点群数別統計
        samples_by_num_pcs = {}
        target_position_stats = {}
        conversation_type_stats = {}
        position_template_stats = {}
        
        for sample in dataset:
            metadata = sample.get("metadata", {})
            num_pcs = metadata.get("num_point_clouds", 0)
            target_pos = metadata.get("target_position", "unknown")
            conv_type = metadata.get("conversation_type", "unknown")
            pos_template = metadata.get("target_position_template", "unknown")
            
            # 点群数別カウント
            key_pcs = f"{num_pcs}_point_clouds"
            samples_by_num_pcs[key_pcs] = samples_by_num_pcs.get(key_pcs, 0) + 1
            
            # ターゲット位置別カウント
            target_position_stats[target_pos] = target_position_stats.get(target_pos, 0) + 1
            
            # 会話タイプ別カウント
            conversation_type_stats[conv_type] = conversation_type_stats.get(conv_type, 0) + 1
            
            # 位置テンプレート別カウント
            position_template_stats[pos_template] = position_template_stats.get(pos_template, 0) + 1
        
        # プロンプト長統計
        prompt_lengths = []
        response_lengths = []
        conversation_turns = []
        
        for sample in dataset:
            conversations = sample.get("conversations", [])
            human_messages = [c for c in conversations if c.get("from") == "human"]
            gpt_messages = [c for c in conversations if c.get("from") == "gpt"]
            
            conversation_turns.append(len(human_messages))
            
            for conv in conversations:
                if conv.get("from") == "human":
                    prompt_lengths.append(len(conv.get("value", "")))
                elif conv.get("from") == "gpt":
                    response_lengths.append(len(conv.get("value", "")))
        
        stats = {
            "total_samples": total_samples,
            "samples_by_num_point_clouds": samples_by_num_pcs,
            "target_position_distribution": target_position_stats,
            "position_template_distribution": position_template_stats,
            "conversation_type_distribution": conversation_type_stats,
            "conversation_statistics": {
                "avg_turns_per_conversation": sum(conversation_turns) / len(conversation_turns) if conversation_turns else 0,
                "max_turns": max(conversation_turns) if conversation_turns else 0,
                "min_turns": min(conversation_turns) if conversation_turns else 0
            },
            "prompt_statistics": {
                "avg_length": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
                "max_length": max(prompt_lengths) if prompt_lengths else 0,
                "min_length": min(prompt_lengths) if prompt_lengths else 0
            },
            "response_statistics": {
                "avg_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
                "max_length": max(response_lengths) if response_lengths else 0,
                "min_length": min(response_lengths) if response_lengths else 0
            },
            "generation_info": {
                "seed": self.seed,
                "complex_data_size": len(self.complex_data),
                "brief_data_size": len(self.brief_data),
                "num_pc_choices": self.num_pc_choices,
                "gpt_api_used": self.use_gpt_api,
                "model_name": self.model_name if self.use_gpt_api else None,
                "api_statistics": self.api_call_stats if self.use_gpt_api else None,
                "generation_type": "complex_instruction_stage2"
            }
        }
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Generate Complex Instruction Stage2 multi-object point cloud dataset")
    parser.add_argument("--data_dir", type=str, default="/groups/gag51404/ide/PointLLM/data/anno_data",
                        help="Directory containing PointLLM JSON files")
    parser.add_argument("--output_dir", type=str, default="/groups/gag51404/ide/PointLLM/data/anno_data",
                        help="Output directory for generated dataset")
    parser.add_argument("--num_samples", type=int, default=70000,
                        help="Number of Complex Stage2 samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_gpt_api", action="store_true",
                        help="Use GPT API for better text rewriting")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (optional, can use OPENAI_API_KEY env var)")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        help="GPT model name to use")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save comparison file showing original vs modified conversations")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Maximum number of worker threads for parallel processing")
    
    args = parser.parse_args()
    
    print("Complex Instruction Stage2 Multi-Object Point Cloud Dataset Generator")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    print(f"GPT API: {'Enabled' if args.use_gpt_api else 'Disabled'}")
    if args.use_gpt_api:
        print(f"Model: {args.model_name}")
    print("Dataset type: Complex instruction with multiple point clouds")
    
    # データセット生成器を作成
    generator = ComplexStage2DatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        use_gpt_api=args.use_gpt_api,
        api_key=args.api_key,
        model_name=args.model_name,
        save_comparison=args.save_comparison,
        max_workers=args.max_workers
    )
    
    # Complex Stage2データセット生成
    dataset = generator.generate_complex_stage2_dataset(num_samples=args.num_samples)
    
    # データセット保存
    output_filename = f"complex_instruction_stage2_multi_pc_70K{'_gpt' if args.use_gpt_api else ''}.json"
    output_path = generator.save_dataset(dataset, filename=output_filename)
    
    print("\nComplex instruction stage2 dataset generation completed!")
    print(f"Output: {output_path}")
    print("\nNext steps:")
    print("1. Use this dataset for Stage2 training with complex instructions")
    print("2. Compare performance with naive baseline")
    print("3. Evaluate model's ability to handle complex multi-turn conversations")
    if args.use_gpt_api:
        print("4. Compare GPT-enhanced dataset quality with regex-based version")

if __name__ == "__main__":
    main() 
    

#python create_complex_stage2_dataset.py --use_gpt_api --num_samples 50 --max_workers 20
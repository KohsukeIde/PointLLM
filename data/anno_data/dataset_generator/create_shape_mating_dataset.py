#!/usr/bin/env python3

"""
Shape Mating Pair Dataset Generator with Object-Level Data Splitting

This script generates a dataset for training models to identify which two parts 
from three given 3D parts form a mating pair, with proper data splitting to prevent
data leakage between training, validation, and test sets.

Key Features:
- Object-level splitting: Ensures the same object never appears in multiple splits
- Three difficulty levels: Easy (different objects) → Medium (same object) → Hard (same object/cut)
- Data leakage prevention: No object overlap between train/val/test splits
- Configurable split ratios: Default 70%/15%/15% for train/val/test
- Support for generating individual splits or all splits at once

Usage Examples:
1. Generate training data only:
   python create_shape_mating_dataset.py --generate_split train --num_samples 50000

2. Generate all splits with custom ratios:
   python create_shape_mating_dataset.py --generate_split all --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

3. Generate test data for evaluation:
   python create_shape_mating_dataset.py --generate_split test --num_samples 5000 --save_split_info
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import re
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI module not available. GPT API features will be disabled.")

class ShapeMatingPairDatasetGenerator:
    """Shape Mating Pair Discovery Dataset Generator - 3つから嵌合ペアを発見するタスク"""
    
    def __init__(self, data_root: str, output_dir: str, seed: int = 42, use_gpt_api: bool = False, 
                 api_key: Optional[str] = None, model_name: str = "gpt-4-turbo", 
                 save_comparison: bool = False, max_workers: int = 10,
                 data_split_ratios: Dict[str, float] = None):
        self.data_root = data_root
        self.output_dir = output_dir
        self.seed = seed
        self.use_gpt_api = use_gpt_api
        self.model_name = model_name
        self.save_comparison = save_comparison
        self.max_workers = max_workers
        
        # データ分割比率の設定
        if data_split_ratios is None:
            self.data_split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        else:
            self.data_split_ratios = data_split_ratios
            
        # 分割比率の検証
        if abs(sum(self.data_split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {sum(self.data_split_ratios.values())}")
        
        # スレッドセーフなランダム生成器
        self.rng = random.Random(seed)
        
        # OpenAI API設定
        if self.use_gpt_api and OPENAI_AVAILABLE:
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                api_key_from_env = os.getenv("OPENAI_API_KEY")
                if not api_key_from_env:
                    print("Warning: No OpenAI API key found. Falling back to non-GPT mode.")
                    self.use_gpt_api = False
                else:
                    self.openai_client = OpenAI(api_key=api_key_from_env)
        else:
            self.use_gpt_api = False
            print("Warning: OpenAI module not available. Falling back to non-GPT mode.")
        
        # 基本プロンプト（タスクタイプ別）- カリキュラム学習順序に修正
        self.base_prompts = {
            1: "Which two of these objects form a matching geometric pair? Please refer to them as 'first', 'second', and 'third' in your answer.",        # Easy - 学習開始
            2: "Identify the pair of components that have complementary surfaces. Please refer to them as 'first', 'second', and 'third' in your answer.", # Medium - 中間段階  
            3: "Among these three parts, which two are designed to fit together? Please refer to them as 'first', 'second', and 'third' in your answer."  # Hard - 最終段階
        }
        
        # 回答テンプレート（多様性確保）
        self.pair_reply_templates = [
            "The {first_ord} and {second_ord} parts fit together.",
            "Parts {first_ord} and {second_ord} form a matching pair.",
            "The {first_ord} and {second_ord} pieces are designed to mate.",
            "Components {first_ord} and {second_ord} complement each other.",
            "The {first_ord} and {second_ord} objects connect perfectly.",
            "Parts {first_ord} and {second_ord} have matching geometries.",
            "The {first_ord} and {second_ord} pieces form a complete unit.",
            "Components {first_ord} and {second_ord} are the mating pair.",
            "The {first_ord} and {second_ord} parts work together.",
            "Parts {first_ord} and {second_ord} are geometrically compatible."
        ]
        
        # プロンプトキャッシュ（スレッドセーフ）
        self.prompt_cache = self._load_prompt_cache()
        self.cache_lock = threading.Lock()
        
        # API統計（スレッドセーフ）
        self.api_call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0
        }
        self.stats_lock = threading.Lock()
        
        # データベース構築とデータ分割
        print("Indexing all generated shape mating parts...")
        self.all_parts_db = self._collect_all_parts()
        print(f"Found {self._count_total_pairs(self.all_parts_db)} valid part pairs.")
        
        # オブジェクトレベルでのデータ分割
        print("Performing object-level data split to prevent data leakage...")
        self.split_assignments = self._create_object_level_split()
        self.train_parts_db, self.val_parts_db, self.test_parts_db = self._split_parts_database()
        
        print(f"Data split: Train={len(self._get_objects_by_split('train'))} objects, "
              f"Val={len(self._get_objects_by_split('val'))} objects, "
              f"Test={len(self._get_objects_by_split('test'))} objects")
        
        # 現在使用中のデータベース（デフォルトはtrain）
        self.current_split = "train"
        self.parts_db = self.train_parts_db
        
    def _count_total_pairs(self, database: Dict = None) -> int:
        """総ペア数をカウント"""
        if database is None:
            database = self.parts_db
            
        total = 0
        for category, objects in database.items():
            for obj_id, cuts in objects.items():
                for cut_type, instances in cuts.items():
                    total += len(instances)
        return total
        
    def _collect_all_parts(self) -> Dict[str, Dict[str, Dict[str, List[Tuple[str, str, str]]]]]:
        """
        全部品データを階層構造で収集
        返り値: {カテゴリ: {オブジェクトID: {切り方: [(インスタンスID, partA_path, partB_path)]}}}
        """
        parts_db = {}
        objaverse_dir = Path(self.data_root) / "objaverse"
        
        if not objaverse_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {objaverse_dir}")
        
        for partA_path in objaverse_dir.rglob('partA-pc.npy'):
            # partB-pc.npyの存在確認
            partB_path = partA_path.parent / 'partB-pc.npy'
            if not partB_path.exists():
                continue
                
            # パス解析（regex with fallback）
            path_str = str(partA_path)
            match = re.search(r'/([^/]+)/shell/([^/]+)/(\d+)/partA-pc\.npy$', path_str)
            
            if match:
                object_id, cut_type, instance_id = match.groups()
            else:
                # フォールバック解析
                parts = partA_path.parts
                if len(parts) >= 4:
                    try:
                        instance_id = parts[-2]
                        cut_type = parts[-3]
                        object_id = parts[-4]
                    except (IndexError, ValueError):
                        continue
                else:
                    continue
            
            # 階層構造に追加
            category = "shell"  # 現在は全てshellカテゴリ
            
            if category not in parts_db:
                parts_db[category] = {}
            if object_id not in parts_db[category]:
                parts_db[category][object_id] = {}
            if cut_type not in parts_db[category][object_id]:
                parts_db[category][object_id][cut_type] = []
                
            parts_db[category][object_id][cut_type].append(
                (instance_id, str(partA_path), str(partB_path))
            )
        
        return parts_db
    
    def _load_prompt_cache(self) -> Dict[str, List[str]]:
        """プロンプトキャッシュを読み込み"""
        cache_path = os.path.join(os.path.dirname(__file__), "prompt_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_prompt_cache(self):
        """プロンプトキャッシュを保存"""
        cache_path = os.path.join(os.path.dirname(__file__), "prompt_cache.json")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.prompt_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save prompt cache: {e}")
    
    def _call_gpt_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """GPT APIを呼び出し（リトライ機能付き、スレッドセーフ）"""
        for attempt in range(max_retries):
            try:
                with self.stats_lock:
                    self.api_call_stats["total_calls"] += 1
                
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
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
    
    def _generate_prompt_variations_with_gpt(self, base_prompt: str, task_type: int, num_variations: int = 30) -> List[str]:
        """GPTを使用してプロンプトのバリエーションを生成"""
        # キャッシュチェック
        cache_key = f"{task_type}_{base_prompt}"
        with self.cache_lock:
            if cache_key in self.prompt_cache:
                return self.prompt_cache[cache_key]
        
        if not self.use_gpt_api:
            return self._generate_prompt_variations_fallback(base_prompt, task_type)
        
        prompt = f"""Generate {num_variations} different ways to ask the same question for a 3D shape mating task where an AI must identify which two parts from three given parts form a mating pair.

Base Question: "{base_prompt}"
Context: This is for Task {task_type} where the AI sees 3 parts and must identify which 2 fit together.

Requirements:
1. Keep the core meaning intact - asking which TWO parts from THREE form a mating pair
2. Use natural, conversational language that humans would use
3. Vary the vocabulary: "fit together", "mate", "pair", "connect", "match", "complement"
4. Vary the referring expressions: "parts", "pieces", "components", "objects"
5. Make it clear that TWO parts need to be identified from THREE total parts
6. Include some questions that mention geometric properties like "surface" or "shape"
7. Make them sound natural for someone examining 3D objects
8. Questions should ask for identifying a PAIR, not a single choice
9. IMPORTANT: Each question must include instruction to refer to parts as 'first', 'second', and 'third' in the answer

Return exactly {num_variations} variations as a JSON list of strings.

Example format:
[
    "Which two of these three parts are designed to fit together?",
    "Identify the pair of components that have matching surfaces.",
    "Among these three objects, which two form a complete assembly?",
    ...
]
"""
        
        result = self._call_gpt_api(prompt)
        
        if result:
            try:
                import json
                variations = json.loads(result)
                if isinstance(variations, list) and len(variations) > 0:
                    with self.cache_lock:
                        self.prompt_cache[cache_key] = variations
                        self._save_prompt_cache()
                    return variations
            except json.JSONDecodeError:
                print(f"Warning: Could not parse GPT response as JSON: {result[:100]}...")
        
        # フォールバックを使用
        with self.stats_lock:
            self.api_call_stats["fallback_used"] += 1
        fallback_variations = self._generate_prompt_variations_fallback(base_prompt, task_type)
        with self.cache_lock:
            self.prompt_cache[cache_key] = fallback_variations
            self._save_prompt_cache()
        return fallback_variations
    
    def _generate_prompt_variations_fallback(self, base_prompt: str, task_type: int) -> List[str]:
        """フォールバック用のプロンプトバリエーション生成（ペア発見タスク用）"""
        
        # ペア発見用のテンプレート
        starters = [
            "Which two of these three parts",
            "Among these three objects, which two", 
            "Identify the pair of components that",
            "From these three pieces, which two",
            "Which two parts from this set",
            "Of the three given parts, which two"
        ]
        
        # 動作を表す動詞句
        verbs = [
            "fit together perfectly?",
            "form a mating pair?", 
            "are designed to connect?",
            "have complementary surfaces?",
            "match geometrically?",
            "work as a matching set?",
            "are meant to be assembled?",
            "create a complete unit?"
        ]
        
        # 追加の表現パターン（序数制約を含む）
        additional_patterns = [
            "Identify the two components that are geometrically compatible. Refer to them as first, second, and third.",
            "Find the pair of parts that connect seamlessly. Please use first, second, and third in your answer.",
            "Which two pieces have surfaces that mate together? Refer to them as first, second, and third.",
            "Determine which two objects form a matching pair. Please refer to them as first, second, and third.",
            "Select the two parts that complement each other. Use first, second, and third in your response.",
            "Point out the two components designed to fit together. Refer to them as first, second, and third.",
            "Choose the two pieces that create a complete assembly. Please use first, second, and third in your answer.",
            "Indicate which two parts have mating surfaces. Refer to them as first, second, and third."
        ]
        
        variations = set()
        
        # 基本パターンの組み合わせ（序数制約を追加）
        ordinal_instructions = [
            " Please refer to them as first, second, and third in your answer.",
            " Use first, second, and third in your response.",
            " Refer to them as first, second, and third.",
            " Please use first, second, and third in your answer."
        ]
        
        for _ in range(50):
            starter = self.rng.choice(starters)
            verb = self.rng.choice(verbs)
            ordinal_inst = self.rng.choice(ordinal_instructions)
            question = starter + " " + verb + ordinal_inst
            variations.add(question)
        
        # 追加パターンを混ぜる
        for pattern in additional_patterns:
            variations.add(pattern)
        
        # リストに変換してシャッフル
        final_variations = list(variations)
        self.rng.shuffle(final_variations)
        
        # 最低限の保証（序数制約を含む）
        if not final_variations:
            return [
                "Which two of these three parts fit together? Please refer to them as first, second, and third.",
                "Identify the pair of components that mate. Use first, second, and third in your answer.",
                "Among these three objects, which two form a matching pair? Refer to them as first, second, and third."
            ]
            
        return final_variations[:30]
    
    def _sample_task1_data(self) -> Tuple[str, str, str, Dict[str, Any]]:
        """タスク1【Easy】: 異種オブジェクト・同一カット - 学習開始レベル"""
        cut_type_objects = {}
        
        for category, objects in self.parts_db.items():
            for obj_id, cuts in objects.items():
                for cut_type in cuts.keys():
                    if cut_type not in cut_type_objects:
                        cut_type_objects[cut_type] = []
                    cut_type_objects[cut_type].append((category, obj_id, cuts[cut_type]))
        
        # オブジェクトが3つ以上あるカットタイプを選択
        multi_object_cuts = {cut: objs for cut, objs in cut_type_objects.items() 
                            if len(objs) >= 3}
        
        if not multi_object_cuts:
            raise ValueError("Not enough data for Task 1 sampling")
        
        # ランダムにカットタイプを選択
        target_cut_type = self.rng.choice(list(multi_object_cuts.keys()))
        available_objects = multi_object_cuts[target_cut_type]
        
        # 3つのオブジェクトをランダムに選択
        selected_objects = self.rng.sample(available_objects, 3)
        
        # 正解ペア（最初のオブジェクト）
        target_category, target_object, target_instances = selected_objects[0]
        target_instance_id, correct_partA, correct_partB = self.rng.choice(target_instances)
        
        # 不正解部品（2番目のオブジェクト）
        negative_category, negative_object, negative_instances = selected_objects[1]
        negative_instance_id, neg_partA, neg_partB = self.rng.choice(negative_instances)
        negative_part = self.rng.choice([neg_partA, neg_partB])
        
        metadata = {
            "task_type": 1,
            "difficulty": "Easy",
            "category": target_category,
            "cut_type": target_cut_type,
            "target_object": target_object,
            "correct_instance": target_instance_id,
            "negative_object": negative_object,
            "negative_instance": negative_instance_id
        }
        
        return correct_partA, correct_partB, negative_part, metadata
    
    def _sample_task2_data(self) -> Tuple[str, str, str, Dict[str, Any]]:
        """タスク2【Medium】: 同一オブジェクト・異種カット - 中間段階"""
        multi_cut_objects = []
        
        for category, objects in self.parts_db.items():
            for obj_id, cuts in objects.items():
                if len(cuts) >= 3:  # カットタイプが3種類以上必要
                    multi_cut_objects.append((category, obj_id, cuts))
        
        if not multi_cut_objects:
            raise ValueError("Not enough data for Task 2 sampling")
        
        # ランダムに選択
        category, target_object, cuts = self.rng.choice(multi_cut_objects)
        
        # 3つのカットタイプをランダムに選択
        cut_types = list(cuts.keys())
        selected_cuts = self.rng.sample(cut_types, 3)
        target_cut_type = selected_cuts[0]
        negative_cut_type = selected_cuts[1]
        
        # 正解ペア
        target_instances = cuts[target_cut_type]
        target_instance_id, correct_partA, correct_partB = self.rng.choice(target_instances)
        
        # 不正解部品
        negative_instances = cuts[negative_cut_type]
        negative_instance_id, neg_partA, neg_partB = self.rng.choice(negative_instances)
        negative_part = self.rng.choice([neg_partA, neg_partB])
        
        metadata = {
            "task_type": 2,
            "difficulty": "Medium",
            "category": category,
            "cut_type": target_cut_type,
            "target_object": target_object,
            "correct_instance": target_instance_id,
            "negative_cut": negative_cut_type,
            "negative_instance": negative_instance_id
        }
        
        return correct_partA, correct_partB, negative_part, metadata
    
    def _sample_task3_data(self) -> Tuple[str, str, str, Dict[str, Any]]:
        """タスク3【Hard】: 同一オブジェクト・同一カット・異種インスタンス - 最高難易度"""
        available_combinations = []
        
        for category, objects in self.parts_db.items():
            for obj_id, cuts in objects.items():
                for cut_type, instances in cuts.items():
                    if len(instances) >= 3:  # インスタンスが3つ以上必要
                        available_combinations.append((category, obj_id, cut_type, instances))
        
        if not available_combinations:
            raise ValueError("Not enough data for Task 3 sampling")
        
        # ランダムに組み合わせを選択
        category, target_object, cut_type, instances = self.rng.choice(available_combinations)
        
        # インスタンスを3つランダムに選ぶ
        selected_instances = self.rng.sample(instances, 3)
        
        # 最初のインスタンスを正解ペアとする
        instance_a_id, correct_partA, correct_partB = selected_instances[0]
        
        # 2番目のインスタンスから不正解部品を選ぶ（partAまたはpartB）
        instance_b_id, neg_partA, neg_partB = selected_instances[1]
        negative_part = self.rng.choice([neg_partA, neg_partB])
        
        metadata = {
            "task_type": 3,
            "difficulty": "Hard",
            "category": category,
            "cut_type": cut_type,
            "target_object": target_object,
            "correct_instance": instance_a_id,
            "negative_instance": instance_b_id
        }
        
        return correct_partA, correct_partB, negative_part, metadata
    
    def _create_shape_mating_sample(self, task_type: int) -> Dict[str, Any]:
        """Shape Mating Pair Discovery用のサンプルを作成"""
        # タスクタイプに応じてデータをサンプリング
        if task_type == 1:
            correct_partA, correct_partB, negative_part, metadata = self._sample_task1_data()
        elif task_type == 2:
            correct_partA, correct_partB, negative_part, metadata = self._sample_task2_data()
        elif task_type == 3:
            correct_partA, correct_partB, negative_part, metadata = self._sample_task3_data()
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        
        # 3つの部品をリストにしてシャッフル
        three_parts = [correct_partA, correct_partB, negative_part]
        self.rng.shuffle(three_parts)
        
        # 正解ペアの位置を記録
        partA_index = three_parts.index(correct_partA)
        partB_index = three_parts.index(correct_partB)
        correct_pair_indices = sorted([partA_index, partB_index])
        
        # プレースホルダーの作成
        identifier_prefix = "<pc_1> <pc_2> <pc_3>"
        
        # 質問文の生成
        base_prompt = self.base_prompts[task_type]
        prompt_variations = self._generate_prompt_variations_with_gpt(base_prompt, task_type)
        instruction = self.rng.choice(prompt_variations)
        
        human_prompt = f"{identifier_prefix}\n{instruction}"
        
        # 回答生成（位置を序数に変換）
        ordinals = ["first", "second", "third"]
        first_ord = ordinals[correct_pair_indices[0]]
        second_ord = ordinals[correct_pair_indices[1]]
        
        reply_template = self.rng.choice(self.pair_reply_templates)
        gpt_response = reply_template.format(first_ord=first_ord, second_ord=second_ord)
        
        # サンプルデータの構築
        sample = {
            "object_ids": three_parts,
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "gpt", 
                    "value": gpt_response
                }
            ],
            "metadata": {
                **metadata,
                "num_point_clouds": 3,
                "correct_pair_indices": correct_pair_indices,
                "correct_pair_positions": [first_ord, second_ord],
                "instruction_variant": instruction,
                "base_prompt": base_prompt,
                "reply_template_used": reply_template,
                "generation_type": f"shape_mating_pair_task{task_type}",
                "gpt_api_used": self.use_gpt_api
            }
        }
        
        return sample
    
    def _process_batch(self, batch_tasks: List[int]) -> List[Dict[str, Any]]:
        """バッチタスクを並列処理"""
        batch_results = []
        for task_type in batch_tasks:
            try:
                result = self._create_shape_mating_sample(task_type)
                batch_results.append(result)
            except Exception as e:
                print(f"Error processing task type {task_type}: {e}")
                continue
        return batch_results
    
    def generate_shape_mating_dataset(self, num_samples: int = 70000, 
                                    task_distribution: Dict[int, float] = None) -> List[Dict[str, Any]]:
        """Shape Mating Pair Discoveryデータセットを生成"""
        if task_distribution is None:
            task_distribution = {1: 0.4, 2: 0.3, 3: 0.3}  # Easy: Medium: Hard = 4:3:3 (カリキュラム学習順序)
        
        print(f"Generating {num_samples} Shape Mating Pair Discovery samples...")
        print(f"Task distribution: Easy={task_distribution[1]:.1f}, Medium={task_distribution[2]:.1f}, Hard={task_distribution[3]:.1f}")
        print(f"Using {self.max_workers} workers for parallel processing")
        
        # タスクタイプのリストを生成
        tasks = []
        for task_type, ratio in task_distribution.items():
            task_count = int(num_samples * ratio)
            tasks.extend([task_type] * task_count)
        
        # 残りをランダムに埋める
        while len(tasks) < num_samples:
            tasks.append(self.rng.choice(list(task_distribution.keys())))
        
        # シャッフル
        self.rng.shuffle(tasks)
        
        # バッチサイズを計算
        batch_size = max(1, len(tasks) // (self.max_workers * 2))
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        print(f"Processing {len(batches)} batches with batch size {batch_size}")
        
        new_dataset = []
        
        # 並列処理実行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
        
        print(f"Generated {len(new_dataset)} Shape Mating Pair Discovery samples")
        
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
                    filename: str = "shape_mating_pair_discovery_70K.json"):
        """データセットを保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"Saving shape mating pair discovery dataset to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # 統計情報を生成・保存
        stats = self._generate_statistics(dataset)
        stats_filename = filename.replace('.json', '_stats.json')
        stats_path = os.path.join(self.output_dir, stats_filename)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Shape mating pair discovery dataset saved to {output_path}")
        print(f"Statistics saved to {stats_path}")
        
        return output_path
    
    def _generate_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """データセット統計を生成"""
        total_samples = len(dataset)
        
        # タスクタイプ別統計
        task_type_stats = {}
        difficulty_stats = {}
        pair_position_stats = {}
        
        for sample in dataset:
            metadata = sample.get("metadata", {})
            task_type = metadata.get("task_type", 0)
            difficulty = metadata.get("difficulty", "unknown")
            pair_indices = metadata.get("correct_pair_indices", [])
            
            # 統計のカウント
            task_type_stats[f"task_{task_type}"] = task_type_stats.get(f"task_{task_type}", 0) + 1
            difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
            
            # ペア位置統計
            if len(pair_indices) == 2:
                pair_key = f"{pair_indices[0]}_{pair_indices[1]}"
                pair_position_stats[pair_key] = pair_position_stats.get(pair_key, 0) + 1
        
        # 回答バリエーション統計
        response_variations = {}
        reply_template_usage = {}
        
        for sample in dataset:
            conversations = sample.get("conversations", [])
            metadata = sample.get("metadata", {})
            
            # 回答テンプレート使用統計
            template_used = metadata.get("reply_template_used", "unknown")
            reply_template_usage[template_used] = reply_template_usage.get(template_used, 0) + 1
            
            for conv in conversations:
                if conv.get("from") == "gpt":
                    response_text = conv.get("value", "")
                    response_variations[response_text] = response_variations.get(response_text, 0) + 1
        
        stats = {
            "total_samples": total_samples,
            "task_type_distribution": task_type_stats,
            "difficulty_distribution": difficulty_stats,
            "pair_position_distribution": pair_position_stats,
            "response_statistics": {
                "unique_responses": len(response_variations),
                "response_diversity_rate": len(response_variations) / total_samples * 100,
                "most_common_responses": dict(sorted(response_variations.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            "reply_template_usage": reply_template_usage,
            "generation_info": {
                "seed": self.seed,
                "data_root": self.data_root,
                "gpt_api_used": self.use_gpt_api,
                "model_name": self.model_name if self.use_gpt_api else None,
                "api_statistics": self.api_call_stats if self.use_gpt_api else None,
                "generation_type": "shape_mating_pair_discovery"
            }
        }
        
        return stats
    
    def _create_object_level_split(self) -> Dict[str, str]:
        """オブジェクトIDレベルでのデータ分割を作成（データリーク防止）"""
        all_object_ids = set()
        for category, objects in self.all_parts_db.items():
            all_object_ids.update(objects.keys())
        
        # オブジェクトIDをソートして決定的な分割を保証
        sorted_object_ids = sorted(list(all_object_ids))
        
        # シャッフル（seedベース）
        shuffled_ids = sorted_object_ids.copy()
        self.rng.shuffle(shuffled_ids)
        
        # 分割点を計算
        total_objects = len(shuffled_ids)
        train_size = int(total_objects * self.data_split_ratios["train"])
        val_size = int(total_objects * self.data_split_ratios["val"])
        
        # 分割を実行
        split_assignments = {}
        
        # Train split
        for i in range(train_size):
            split_assignments[shuffled_ids[i]] = "train"
        
        # Validation split
        for i in range(train_size, train_size + val_size):
            split_assignments[shuffled_ids[i]] = "val"
        
        # Test split (残り全て)
        for i in range(train_size + val_size, total_objects):
            split_assignments[shuffled_ids[i]] = "test"
        
        return split_assignments
    
    def _split_parts_database(self) -> Tuple[Dict, Dict, Dict]:
        """parts_databaseを分割済みデータベースに分割"""
        train_db = {}
        val_db = {}
        test_db = {}
        
        for category, objects in self.all_parts_db.items():
            train_db[category] = {}
            val_db[category] = {}
            test_db[category] = {}
            
            for obj_id, cuts in objects.items():
                split = self.split_assignments[obj_id]
                
                if split == "train":
                    train_db[category][obj_id] = cuts
                elif split == "val":
                    val_db[category][obj_id] = cuts
                elif split == "test":
                    test_db[category][obj_id] = cuts
        
        return train_db, val_db, test_db
    
    def _get_objects_by_split(self, split_name: str) -> List[str]:
        """指定されたsplitのオブジェクトIDリストを取得"""
        return [obj_id for obj_id, split in self.split_assignments.items() if split == split_name]
    
    def set_split(self, split_name: str):
        """現在使用するデータ分割を設定"""
        if split_name not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split name: {split_name}. Must be 'train', 'val', or 'test'")
        
        self.current_split = split_name
        if split_name == "train":
            self.parts_db = self.train_parts_db
        elif split_name == "val":
            self.parts_db = self.val_parts_db
        elif split_name == "test":
            self.parts_db = self.test_parts_db
        
        print(f"Switched to {split_name} split with {len(self._get_objects_by_split(split_name))} objects")
        
    def get_split_info(self) -> Dict[str, Any]:
        """データ分割の情報を取得"""
        info = {
            "current_split": self.current_split,
            "split_ratios": self.data_split_ratios,
            "object_counts": {
                "train": len(self._get_objects_by_split("train")),
                "val": len(self._get_objects_by_split("val")),
                "test": len(self._get_objects_by_split("test"))
            },
            "total_objects": len(self.split_assignments)
        }
        
        # 各splitのペア数も計算
        info["train_pairs"] = self._count_total_pairs(self.train_parts_db)
        info["val_pairs"] = self._count_total_pairs(self.val_parts_db)
        info["test_pairs"] = self._count_total_pairs(self.test_parts_db)
        
        return info

def main():
    parser = argparse.ArgumentParser(description="Generate Shape Mating Pair Discovery dataset")
    parser.add_argument("--data_root", type=str, default="/groups/gag51402/datasets/ORB3D/objaverse_PWN_filtered",
                        help="Root directory containing shape data")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory for generated dataset")
    parser.add_argument("--num_samples", type=int, default=70000,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_gpt_api", action="store_true",
                        help="Use GPT API for better prompt generation")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (optional, can use OPENAI_API_KEY env var)")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="GPT model name to use")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save comparison file showing prompt variations")
    parser.add_argument("--max_workers", type=int, default=150,
                        help="Maximum number of worker threads for parallel processing")
    
    # データ分割関連の引数
    parser.add_argument("--generate_split", type=str, choices=["train", "val", "test", "all"], default="train",
                        help="Which data split to generate (train/val/test/all)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Training data ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation data ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Test data ratio (default: 0.15)")
    parser.add_argument("--save_split_info", action="store_true",
                        help="Save detailed split information")
    
    args = parser.parse_args()
    
    # データ分割比率の検証
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Split ratios must sum to 1.0, got {total_ratio}")
        return
    
    data_split_ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio
    }
    
    print("Shape Mating Pair Discovery Dataset Generator")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    print(f"GPT API: {'Enabled' if args.use_gpt_api else 'Disabled'}")
    if args.use_gpt_api:
        print(f"Model: {args.model_name}")
    print("Dataset type: Shape Mating Pair Discovery (3 parts → identify which 2 mate)")
    print(f"Generate split: {args.generate_split}")
    print(f"Split ratios: Train={args.train_ratio:.2f}, Val={args.val_ratio:.2f}, Test={args.test_ratio:.2f}")
    
    # データセット生成器を作成
    generator = ShapeMatingPairDatasetGenerator(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        use_gpt_api=args.use_gpt_api,
        api_key=args.api_key,
        model_name=args.model_name,
        save_comparison=args.save_comparison,
        max_workers=args.max_workers,
        data_split_ratios=data_split_ratios
    )
    
    # 分割情報を表示
    split_info = generator.get_split_info()
    print(f"\nData Split Information:")
    print(f"  Total objects: {split_info['total_objects']}")
    print(f"  Train: {split_info['object_counts']['train']} objects ({split_info['train_pairs']} pairs)")
    print(f"  Val: {split_info['object_counts']['val']} objects ({split_info['val_pairs']} pairs)")
    print(f"  Test: {split_info['object_counts']['test']} objects ({split_info['test_pairs']} pairs)")
    
    # 分割情報を保存（オプション）
    if args.save_split_info:
        split_info_path = os.path.join(args.output_dir, "data_split_info.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(split_info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"Split information saved to: {split_info_path}")
    
    # データセット生成
    if args.generate_split == "all":
        # 全splitのデータセットを生成
        for split_name in ["train", "val", "test"]:
            print(f"\n{'='*50}")
            print(f"Generating {split_name.upper()} dataset...")
            print(f"{'='*50}")
            
            generator.set_split(split_name)
            
            # 各splitに応じたサンプル数を計算
            if split_name == "train":
                split_samples = args.num_samples
            elif split_name == "val":
                split_samples = max(1000, int(args.num_samples * 0.1))  # 10%、最低1000サンプル
            else:  # test
                split_samples = max(2000, int(args.num_samples * 0.2))  # 20%、最低2000サンプル
            
            dataset = generator.generate_shape_mating_dataset(num_samples=split_samples)
            
            # 保存
            output_filename = f"shape_mating_pair_discovery_{split_name}_{split_samples//1000}K{'_gpt' if args.use_gpt_api else ''}.json"
            output_path = generator.save_dataset(dataset, filename=output_filename)
            print(f"{split_name.upper()} dataset saved to: {output_path}")
    
    else:
        # 指定されたsplitのみ生成
        generator.set_split(args.generate_split)
        dataset = generator.generate_shape_mating_dataset(num_samples=args.num_samples)
        
        # データセット保存
        output_filename = f"shape_mating_pair_discovery_{args.generate_split}_{args.num_samples//1000}K{'_gpt' if args.use_gpt_api else ''}.json"
        output_path = generator.save_dataset(dataset, filename=output_filename)
        print(f"\n{args.generate_split.upper()} dataset saved to: {output_path}")
    
    print("\nShape Mating Pair Discovery dataset generation completed!")
    print("\n🎯 Key Features:")
    print("✅ Pair Discovery Task: 3 parts → identify which 2 mate")
    print("✅ Difficulty Levels: Easy (diff obj) → Medium (same obj) → Hard (same obj/cut)")
    print("✅ Balanced Distribution: 40% Easy, 30% Medium, 30% Hard")
    print("✅ Response Diversity: 10+ different reply templates")
    print("✅ Thread Safety: Parallel processing with proper synchronization")
    print("✅ Evaluation Ready: correct_pair_indices for automated evaluation")
    print("✅ Data Split Protection: Object-level split prevents data leakage")
    print(f"✅ Split Configuration: {data_split_ratios}")

if __name__ == "__main__":
    main() 
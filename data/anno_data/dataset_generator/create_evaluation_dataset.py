#!/usr/bin/env python3

import os
import json
import random
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

class EvaluationDatasetGenerator:
    """評価用の複数点群データセット生成器"""
    
    def __init__(self, data_dir: str, output_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        
        random.seed(seed)
        
        # 指示文テンプレート
        self.templates = {
            "first": [
                "Summarize the first 3D point cloud object briefly.",
                "What kind of object is depicted by the first point cloud?",
                "Provide a short explanation of the first 3D structure.",
                "What does the first collection of points represent?",
                "Offer a succinct summary of the first 3D object.",
                "Can you give a brief overview of the first point cloud?",
                "Characterize the object the first point cloud is illustrating.",
                "Share a brief interpretation of the first 3D point cloud.",
                "Provide an outline of the first 3D shape's characteristics.",
                "What object is the first point cloud rendering?"
            ],
            "second": [
                "Summarize the second 3D point cloud object briefly.",
                "What kind of object is depicted by the second point cloud?",
                "Provide a short explanation of the second 3D structure.",
                "What does the second collection of points represent?",
                "Offer a succinct summary of the second 3D object.",
                "Can you give a brief overview of the second point cloud?",
                "Characterize the object the second point cloud is illustrating.",
                "Share a brief interpretation of the second 3D point cloud.",
                "Provide an outline of the second 3D shape's characteristics.",
                "What object is the second point cloud rendering?"
            ],
            "third": [
                "Summarize the third 3D point cloud object briefly.",
                "What kind of object is depicted by the third point cloud?",
                "Provide a short explanation of the third 3D structure.",
                "What does the third collection of points represent?",
                "Offer a succinct summary of the third 3D object.",
                "Can you give a brief overview of the third point cloud?",
                "Characterize the object the third point cloud is illustrating.",
                "Share a brief interpretation of the third 3D point cloud.",
                "Provide an outline of the third 3D shape's characteristics.",
                "What object is the third point cloud rendering?"
            ]
        }
        
        # 1サンプルあたりに含める点群の数の選択肢
        self.num_pc_choices = [2, 3]
    
    def _load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """JSONデータを読み込む"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _extract_gpt_response(self, sample: Dict[str, Any]) -> str:
        """サンプルからGPTの応答を抽出"""
        for conv in sample.get("conversations", []):
            if conv.get("from") == "gpt":
                return conv.get("value", "").strip()
        return "unknown object"
    
    def _create_evaluation_sample(self, source_samples: List[Dict[str, Any]], target_index: int) -> Dict[str, Any]:
        """評価用のサンプルを作成"""
        num_pcs = len(source_samples)
        
        # オブジェクトIDを取得
        object_ids = []
        for sample in source_samples:
            if isinstance(sample.get("object_id"), list):
                object_ids.append(sample["object_id"][0])
            else:
                object_ids.append(sample.get("object_id", "unknown"))
        
        # 指示文を生成
        position_keys = ["first", "second", "third"]
        position_key = position_keys[target_index]
        instruction_template = random.choice(self.templates[position_key])
        
        # 識別子プレースホルダーの生成
        pc_identifiers = [f'<pc_{i+1}>' for i in range(num_pcs)]
        identifier_prefix = ' '.join(pc_identifiers)
        
        final_instruction = f"{identifier_prefix}\n{instruction_template}"
        
        # 正解応答（ターゲット点群の元のキャプション）
        correct_response = self._extract_gpt_response(source_samples[target_index])
        
        # 新しいサンプルを構築
        new_sample = {
            "object_ids": object_ids,  # 複数のIDをリストで保持
            "conversation_type": "complex_instruction",  # データローダーのフィルタリング用
            "conversations": [
                {
                    "from": "human",
                    "value": final_instruction
                },
                {
                    "from": "gpt",
                    "value": correct_response
                }
            ],
            # デバッグ用の追加情報
            "metadata": {
                "num_point_clouds": num_pcs,
                "target_index": target_index,
                "target_position": position_key,
                "source_sample_ids": [sample.get("object_id", "unknown") for sample in source_samples],
                "eval_type": "multi_pc_evaluation"
            }
        }
        
        return new_sample
    
    def generate_evaluation_dataset(self, input_file: str, num_samples: int = 200) -> List[Dict[str, Any]]:
        """評価用データセットを生成"""
        # 入力データを読み込み
        input_path = os.path.join(self.data_dir, input_file)
        original_data = self._load_json_data(input_path)
        
        if len(original_data) < 3:
            raise ValueError("Need at least 3 samples in original data to generate evaluation dataset")
        
        print(f"Generating {num_samples} evaluation samples...")
        print(f"Source data size: {len(original_data)}")
        
        new_dataset = []
        
        for i in tqdm(range(num_samples), desc="Generating Evaluation Data"):
            # 1. 1サンプルあたりの点群数をランダムに決定 (2 or 3)
            num_pcs = random.choice(self.num_pc_choices)
            
            # 2. 元データから点群数分のサンプルをランダムに抽出
            source_samples = random.sample(original_data, num_pcs)
            
            # 3. どの点群について質問するか、ターゲットをランダムに決定 (0-indexed)
            target_index = random.randint(0, num_pcs - 1)
            
            # 4. 評価用サンプルを作成
            eval_sample = self._create_evaluation_sample(source_samples, target_index)
            new_dataset.append(eval_sample)
        
        print(f"Generated {len(new_dataset)} evaluation samples")
        return new_dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """データセットを保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"Saving evaluation dataset to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # 統計情報を生成・保存
        stats = self._generate_statistics(dataset)
        stats_filename = filename.replace('.json', '_stats.json')
        stats_path = os.path.join(self.output_dir, stats_filename)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation dataset saved to {output_path}")
        print(f"Statistics saved to {stats_path}")
        
        return output_path
    
    def _generate_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """データセット統計を生成"""
        total_samples = len(dataset)
        
        # 点群数別統計
        samples_by_num_pcs = {}
        target_position_stats = {}
        
        for sample in dataset:
            metadata = sample.get("metadata", {})
            num_pcs = metadata.get("num_point_clouds", 0)
            target_pos = metadata.get("target_position", "unknown")
            
            # 点群数別カウント
            key_pcs = f"{num_pcs}_point_clouds"
            samples_by_num_pcs[key_pcs] = samples_by_num_pcs.get(key_pcs, 0) + 1
            
            # ターゲット位置別カウント
            target_position_stats[target_pos] = target_position_stats.get(target_pos, 0) + 1
        
        stats = {
            "total_samples": total_samples,
            "samples_by_num_point_clouds": samples_by_num_pcs,
            "target_position_distribution": target_position_stats,
            "generation_info": {
                "seed": self.seed,
                "num_pc_choices": self.num_pc_choices,
                "eval_type": "multi_pc_evaluation",
                "purpose": "checkpoint evaluation"
            }
        }
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation dataset for multi-object point cloud models")
    parser.add_argument("--data_dir", type=str, default="/groups/gag51404/ide/PointLLM/data/anno_data",
                        help="Directory containing input JSON files")
    parser.add_argument("--output_dir", type=str, default="/groups/gag51404/ide/PointLLM/data/anno_data",
                        help="Output directory for generated dataset")
    parser.add_argument("--input_file", type=str, default="PointLLM_brief_description_val_200_GT.json",
                        help="Input validation file")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of evaluation samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("Multi-Object Point Cloud Evaluation Dataset Generator")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Input file: {args.input_file}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    
    # データセット生成器を作成
    generator = EvaluationDatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # 評価用データセット生成
    dataset = generator.generate_evaluation_dataset(
        input_file=args.input_file,
        num_samples=args.num_samples
    )
    
    # ファイル名を生成
    base_name = os.path.splitext(args.input_file)[0]
    output_filename = f"{base_name}_multi_pc_eval_{args.num_samples}.json"
    
    # データセット保存
    output_path = generator.save_dataset(dataset, output_filename)
    
    print("\nEvaluation dataset generation completed!")
    print(f"Output: {output_path}")
    print("\nNext steps:")
    print("1. Use this dataset with eval_multi_pc.py script")
    print("2. Run evaluation with checkpoint models")
    print("3. Compare results between different training stages")

if __name__ == "__main__":
    main() 
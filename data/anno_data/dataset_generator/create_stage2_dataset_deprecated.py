#!/usr/bin/env python3

import os
import json
import random
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

class Stage2DatasetGenerator:
    """Stage2用の複数点群データセット生成器"""
    
    def __init__(self, data_dir: str, output_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        
        random.seed(seed)
        
        # 入力データファイルのパス
        self.input_file = os.path.join(data_dir, "PointLLM_brief_description_660K_filtered.json")
        
        # データ読み込み
        self.original_data = self._load_json_data(self.input_file)
        
        print(f"Loaded {len(self.original_data)} original samples")
        
        # 指示文テンプレート（テンプレートファイルから継承）
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
                "What object is the first point cloud rendering?",
                "Deliver a quick description of the object represented in the first point cloud.",
                "How would you describe the 3D form shown in the first point cloud?",
                "What is the nature of the object the first point cloud is representing?",
                "Present a compact account of the first 3D object's key features.",
                "What can you infer about the object from the first point cloud?",
                "Offer a clear and concise description of the first point cloud object.",
                "How would you summarize the first 3D data set?",
                "Give a brief explanation of the object that the first cloud of points forms.",
                "What kind of structure does the first 3D point cloud depict?",
                "Could you delineate the form indicated by the first point cloud?",
                "Express in brief, what the first point cloud is representing.",
                "Give a quick overview of the object represented by the first 3D cloud.",
                "Convey a summary of the 3D structure represented in the first point cloud.",
                "What kind of object is illustrated by the first collection of points?",
                "Describe the object that the first point cloud forms.",
                "How would you interpret the first 3D point cloud?",
                "Can you briefly outline the shape represented by the points in the first point cloud?",
                "Give a concise interpretation of the 3D data presented in the first point cloud.",
                "Explain the object the first point cloud depicts succinctly.",
                "Offer a summary of the 3D object illustrated by the first cloud."
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
                "What object is the second point cloud rendering?",
                "Deliver a quick description of the object represented in the second point cloud.",
                "How would you describe the 3D form shown in the second point cloud?",
                "What is the nature of the object the second point cloud is representing?",
                "Present a compact account of the second 3D object's key features.",
                "What can you infer about the object from the second point cloud?",
                "Offer a clear and concise description of the second point cloud object.",
                "How would you summarize the second 3D data set?",
                "Give a brief explanation of the object that the second cloud of points forms.",
                "What kind of structure does the second 3D point cloud depict?",
                "Could you delineate the form indicated by the second point cloud?",
                "Express in brief, what the second point cloud is representing.",
                "Give a quick overview of the object represented by the second 3D cloud.",
                "Convey a summary of the 3D structure represented in the second point cloud.",
                "What kind of object is illustrated by the second collection of points?",
                "Describe the object that the second point cloud forms.",
                "How would you interpret the second 3D point cloud?",
                "Can you briefly outline the shape represented by the points in the second point cloud?",
                "Give a concise interpretation of the 3D data presented in the second point cloud.",
                "Explain the object the second point cloud depicts succinctly.",
                "Offer a summary of the 3D object illustrated by the second cloud."
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
                "What object is the third point cloud rendering?",
                "Deliver a quick description of the object represented in the third point cloud.",
                "How would you describe the 3D form shown in the third point cloud?",
                "What is the nature of the object the third point cloud is representing?",
                "Present a compact account of the third 3D object's key features.",
                "What can you infer about the object from the third point cloud?",
                "Offer a clear and concise description of the third point cloud object.",
                "How would you summarize the third 3D data set?",
                "Give a brief explanation of the object that the third cloud of points forms.",
                "What kind of structure does the third 3D point cloud depict?",
                "Could you delineate the form indicated by the third point cloud?",
                "Express in brief, what the third point cloud is representing.",
                "Give a quick overview of the object represented by the third 3D cloud.",
                "Convey a summary of the 3D structure represented in the third point cloud.",
                "What kind of object is illustrated by the third collection of points?",
                "Describe the object that the third point cloud forms.",
                "How would you interpret the third 3D point cloud?",
                "Can you briefly outline the shape represented by the points in the third point cloud?",
                "Give a concise interpretation of the 3D data presented in the third point cloud.",
                "Explain the object the third point cloud depicts succinctly.",
                "Offer a summary of the 3D object illustrated by the third cloud."
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
    
    def _create_stage2_sample(self, source_samples: List[Dict[str, Any]], target_index: int) -> Dict[str, Any]:
        """Stage2用のサンプルを作成"""
        num_pcs = len(source_samples)
        
        # オブジェクトIDを取得
        object_ids = []
        for sample in source_samples:
            if isinstance(sample.get("object_id"), list):
                # 既に複数オブジェクトの場合は最初のIDを使用
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
                "source_sample_ids": [sample.get("object_id", "unknown") for sample in source_samples]
            }
        }
        
        return new_sample
    
    def generate_stage2_dataset(self, num_samples: int = 70000) -> List[Dict[str, Any]]:
        """Stage2用のデータセットを生成"""
        if len(self.original_data) < 3:
            raise ValueError("Need at least 3 samples in original data to generate stage2 dataset")
        
        print(f"Generating {num_samples} Stage2 samples...")
        print(f"Source data size: {len(self.original_data)}")
        
        new_dataset = []
        
        for i in tqdm(range(num_samples), desc="Generating Stage2 Data"):
            # 1. 1サンプルあたりの点群数をランダムに決定 (2 or 3)
            num_pcs = random.choice(self.num_pc_choices)
            
            # 2. 元データから点群数分のサンプルをランダムに抽出
            source_samples = random.sample(self.original_data, num_pcs)
            
            # 3. どの点群について質問するか、ターゲットをランダムに決定 (0-indexed)
            target_index = random.randint(0, num_pcs - 1)
            
            # 4. Stage2サンプルを作成
            stage2_sample = self._create_stage2_sample(source_samples, target_index)
            new_dataset.append(stage2_sample)
        
        print(f"Generated {len(new_dataset)} Stage2 samples")
        return new_dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "PointLLM_complex_instruction_stage2_multi_pc_70K.json"):
        """データセットを保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"Saving dataset to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # 統計情報を生成・保存
        stats = self._generate_statistics(dataset)
        stats_path = os.path.join(self.output_dir, "PointLLM_complex_instruction_stage2_multi_pc_70K_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {output_path}")
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
        
        # プロンプト長統計
        prompt_lengths = []
        response_lengths = []
        
        for sample in dataset:
            conversations = sample.get("conversations", [])
            if len(conversations) >= 2:
                human_msg = conversations[0]["value"]
                gpt_msg = conversations[1]["value"]
                prompt_lengths.append(len(human_msg))
                response_lengths.append(len(gpt_msg))
        
        stats = {
            "total_samples": total_samples,
            "samples_by_num_point_clouds": samples_by_num_pcs,
            "target_position_distribution": target_position_stats,
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
                "source_data_size": len(self.original_data),
                "num_pc_choices": self.num_pc_choices,
                "template_counts": {
                    "first_templates": len(self.templates["first"]),
                    "second_templates": len(self.templates["second"]),
                    "third_templates": len(self.templates["third"])
                }
            }
        }
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Generate Stage2 multi-object point cloud dataset")
    parser.add_argument("--data_dir", type=str, default="/groups/gag51404/ide/PointLLM/data/anno_data",
                        help="Directory containing PointLLM JSON files")
    parser.add_argument("--output_dir", type=str, default="/groups/gag51404/ide/PointLLM/data/anno_data",
                        help="Output directory for generated dataset")
    parser.add_argument("--num_samples", type=int, default=70000,
                        help="Number of Stage2 samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("Stage2 Multi-Object Point Cloud Dataset Generator")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    
    # データセット生成器を作成
    generator = Stage2DatasetGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Stage2データセット生成
    dataset = generator.generate_stage2_dataset(num_samples=args.num_samples)
    
    # データセット保存
    output_path = generator.save_dataset(dataset)
    
    print("\nStage2 dataset generation completed!")
    print(f"Output: {output_path}")
    print("\nNext steps:")
    print("1. Modify data loader to handle multiple object_ids")
    print("2. Create training script for Stage2 with this dataset")

if __name__ == "__main__":
    main() 
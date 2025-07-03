import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
import random

class Cap3DOrdinalDataset(Dataset):
    """Cap3D序数参照データセット用のDatasetクラス"""
    
    def __init__(self, dataset_file: str, objaverse_data_dir: str = "/groups/gag51402/datasets/objaverse_data"):
        self.dataset_file = dataset_file
        self.objaverse_data_dir = objaverse_data_dir
        self.objaverse_npy_dir = os.path.join(objaverse_data_dir, "8192_npy")
        
        # データセット読み込み
        with open(dataset_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} Cap3D ordinal samples from {dataset_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 点群データを読み込み
        point_clouds = []
        object_ids = sample["object_id"]
        
        # object_idがリストでない場合はリストに変換
        if not isinstance(object_ids, list):
            object_ids = [object_ids]
        
        for obj_id in object_ids:
            npy_path = os.path.join(self.objaverse_npy_dir, f"{obj_id}_8192.npy")
            if os.path.exists(npy_path):
                points = np.load(npy_path)  # (8192, 6) - xyz + rgb
                # bfloat16対応のためfloat32で統一
                point_clouds.append(torch.from_numpy(points.astype(np.float32)))
            else:
                print(f"Warning: Point cloud file not found: {npy_path}")
                # ダミーデータを使用
                dummy_points = np.random.randn(8192, 6).astype(np.float32)
                point_clouds.append(torch.from_numpy(dummy_points))
        
        # 点群データを結合 (M, N, C)形式
        if len(point_clouds) > 1:
            point_clouds_tensor = torch.stack(point_clouds, dim=0)  # (M, N, C)
        else:
            point_clouds_tensor = point_clouds[0].unsqueeze(0)  # (1, N, C)
        
        # プロンプトと回答を抽出
        conversations = sample["conversations"]
        prompt = None
        answer = None
        
        for conv in conversations:
            if conv["from"] == "human":
                prompt = conv["value"]
            elif conv["from"] == "gpt":
                answer = conv["value"]
        
        return {
            'sample_id': sample.get("sample_id", f"sample_{idx}"),
            'object_ids': object_ids,
            'num_objects': sample["num_objects"],
            'point_clouds': point_clouds_tensor,
            'prompt': prompt,
            'answer': answer,
            'head_nouns': sample.get("head_nouns", [])
        }

def create_cap3d_dataloader(dataset_file: str, 
                           objaverse_data_dir: str = "/groups/gag51402/datasets/objaverse_data",
                           batch_size: int = 1,
                           shuffle: bool = True,
                           num_workers: int = 0) -> DataLoader:
    """Cap3D序数参照データ用のDataLoaderを作成"""
    
    dataset = Cap3DOrdinalDataset(
        dataset_file=dataset_file,
        objaverse_data_dir=objaverse_data_dir
    )
    
    def collate_fn(batch):
        """カスタムバッチ処理関数"""
        if len(batch) == 1:
            # バッチサイズ1の場合は直接返す
            return batch[0]
        else:
            # 複数サンプルのバッチ処理
            batch_data = {
                'sample_id': [item['sample_id'] for item in batch],
                'object_ids': [item['object_ids'] for item in batch],
                'num_objects': [item['num_objects'] for item in batch],
                'point_clouds': [item['point_clouds'] for item in batch],  # リストのまま保持
                'prompt': [item['prompt'] for item in batch],
                'answer': [item['answer'] for item in batch],
                'head_nouns': [item['head_nouns'] for item in batch]
            }
            return batch_data
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader

def analyze_cap3d_dataset(dataset_file: str):
    """Cap3Dデータセットの分析"""
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 50)
    print("Cap3D Ordinal Dataset Analysis")
    print("=" * 50)
    
    print(f"Total Samples: {len(data)}")
    
    # オブジェクト数別統計
    obj_count_stats = {}
    prompt_stats = {}
    
    for sample in data:
        num_obj = sample["num_objects"]
        if num_obj not in obj_count_stats:
            obj_count_stats[num_obj] = 0
        obj_count_stats[num_obj] += 1
        
        # プロンプト統計
        for conv in sample["conversations"]:
            if conv["from"] == "human":
                prompt = conv["value"]
                if prompt not in prompt_stats:
                    prompt_stats[prompt] = 0
                prompt_stats[prompt] += 1
    
    print("\nSamples by Object Count:")
    for num_obj, count in sorted(obj_count_stats.items()):
        print(f"  {num_obj} objects: {count} samples")
    
    print("\nPrompt Distribution:")
    for prompt, count in sorted(prompt_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{prompt}': {count} samples")
    
    # サンプル表示
    print("\nSample Examples:")
    for i in range(min(3, len(data))):
        sample = data[i]
        print(f"\nSample {i+1}:")
        print(f"  ID: {sample.get('sample_id', 'N/A')}")
        print(f"  Objects: {sample['num_objects']}")
        print(f"  Object IDs: {sample['object_id']}")
        print(f"  Head Nouns: {sample.get('head_nouns', 'N/A')}")
        
        for conv in sample["conversations"]:
            print(f"  {conv['from']}: {conv['value']}")

if __name__ == "__main__":
    # テスト用
    dataset_file = "data/cap3d_ordinal_training/ordinal_dataset_cap3d.json"
    if os.path.exists(dataset_file):
        analyze_cap3d_dataset(dataset_file)
        
        # データローダーテスト
        dataloader = create_cap3d_dataloader(dataset_file, batch_size=1, shuffle=False)
        
        print("\nDataLoader Test:")
        for i, batch in enumerate(dataloader):
            if i >= 2:  # 最初の2サンプルのみテスト
                break
            print(f"Batch {i+1}:")
            print(f"  Sample ID: {batch['sample_id']}")
            print(f"  Point Clouds Shape: {batch['point_clouds'].shape}")
            print(f"  Prompt: {batch['prompt']}")
            print(f"  Answer: {batch['answer']}")
    else:
        print(f"Dataset file not found: {dataset_file}") 
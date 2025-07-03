#!/usr/bin/env python3
"""
第1段階学習用データセット作成スクリプト

目的: 複数の識別トークン（<pcA>, <pcB>, <pcC>）が
     すべて「点群を表すプレースホルダー」という共通の役割を持つことを学習させる

処理: 元データセットの<point>を3種類の識別トークンからランダムに1つ選んで置き換え
"""

import json
import random
import os
from tqdm import tqdm
import argparse
from typing import List, Dict, Any

class Stage1DatasetCreator:
    """第1段階学習用データセット作成器"""
    
    def __init__(self, input_file: str, output_file: str, seed: int = 42):
        self.input_file = input_file
        self.output_file = output_file
        self.seed = seed
        
        # ランダムシード設定
        random.seed(seed)
        
        # 識別トークン定義（Phase 2で使用するものと同じ）
        self.identifiers = ['<pc_1>', '<pc_2>', '<pc_3>']
        
        # 元データで使用されているプレースホルダー
        self.original_placeholder = '<point>'
        
        print(f"Stage 1 Dataset Creator初期化")
        print(f"入力ファイル: {input_file}")
        print(f"出力ファイル: {output_file}")
        print(f"識別トークン: {self.identifiers}")
        print(f"ランダムシード: {seed}")
    
    def load_original_dataset(self) -> List[Dict[str, Any]]:
        """元のデータセットを読み込み"""
        print(f"\nデータセット読み込み中: {self.input_file}")
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"入力ファイルが見つかりません: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"読み込み完了: {len(data)} サンプル")
        return data
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        1つのサンプルを処理して識別トークンを割り当て
        
        Args:
            sample: 元のサンプルデータ
            
        Returns:
            処理済みサンプル（識別トークン付き）
        """
        # 識別トークンをランダムに選択
        chosen_identifier = random.choice(self.identifiers)
        
        # 元のサンプルをディープコピー
        new_sample = sample.copy()
        if 'conversations' in new_sample:
            new_sample['conversations'] = [conv.copy() for conv in sample['conversations']]
        
        # conversationsリストを処理
        modified = False
        for i, conversation in enumerate(new_sample.get('conversations', [])):
            if conversation.get('from') == 'human':
                original_value = conversation.get('value', '')
                if self.original_placeholder in original_value:
                    # <point>を選択された識別トークンに置き換え
                    new_value = original_value.replace(self.original_placeholder, chosen_identifier)
                    new_sample['conversations'][i]['value'] = new_value
                    modified = True
        
        # 処理情報をメタデータとして追加
        new_sample['stage1_metadata'] = {
            'assigned_identifier': chosen_identifier,
            'original_placeholder': self.original_placeholder,
            'modified': modified
        }
        
        return new_sample
    
    def create_dataset(self) -> List[Dict[str, Any]]:
        """新しいデータセットを作成"""
        # 元データセット読み込み
        original_data = self.load_original_dataset()
        
        print(f"\nデータセット処理開始...")
        
        new_data_list = []
        skipped_count = 0
        identifier_counts = {token: 0 for token in self.identifiers}
        
        for sample in tqdm(original_data, desc="サンプル処理中"):
            try:
                # サンプル処理
                new_sample = self.process_sample(sample)
                
                # 統計更新
                assigned_token = new_sample['stage1_metadata']['assigned_identifier']
                identifier_counts[assigned_token] += 1
                
                # 処理されたサンプルを追加
                if new_sample['stage1_metadata']['modified']:
                    new_data_list.append(new_sample)
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"警告: object_id {sample.get('object_id', 'unknown')} の処理でエラー: {e}")
                skipped_count += 1
        
        # 処理結果の統計表示
        print(f"\n処理完了:")
        print(f"  処理済みサンプル: {len(new_data_list)}")
        print(f"  スキップしたサンプル: {skipped_count}")
        print(f"  識別トークン分布:")
        for token, count in identifier_counts.items():
            percentage = (count / len(new_data_list)) * 100 if new_data_list else 0
            print(f"    {token}: {count} ({percentage:.1f}%)")
        
        return new_data_list
    
    def save_dataset(self, data_list: List[Dict[str, Any]]):
        """新しいデータセットを保存"""
        print(f"\nデータセット保存中: {self.output_file}")
        
        # 出力ディレクトリを作成
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # JSONファイルとして保存
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"保存完了: {len(data_list)} サンプル")
        
        # 統計ファイルも生成
        stats_file = self.output_file.replace('.json', '_stats.json')
        self.save_statistics(data_list, stats_file)
    
    def save_statistics(self, data_list: List[Dict[str, Any]], stats_file: str):
        """データセット統計を保存"""
        # 識別トークン分布
        identifier_counts = {token: 0 for token in self.identifiers}
        for sample in data_list:
            token = sample.get('stage1_metadata', {}).get('assigned_identifier')
            if token in identifier_counts:
                identifier_counts[token] += 1
        
        # 統計情報
        stats = {
            'total_samples': len(data_list),
            'identifiers_used': self.identifiers,
            'original_placeholder': self.original_placeholder,
            'identifier_distribution': identifier_counts,
            'identifier_percentages': {
                token: (count / len(data_list)) * 100 if data_list else 0
                for token, count in identifier_counts.items()
            },
            'generation_config': {
                'seed': self.seed,
                'input_file': self.input_file,
                'output_file': self.output_file
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"統計情報保存: {stats_file}")
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """サンプルの妥当性をチェック"""
        # 基本構造チェック
        if 'conversations' not in sample:
            return False
        
        # conversationsが空でないことをチェック
        if not sample['conversations']:
            return False
        
        # 最初のconversationがhumanからのものかチェック
        first_conv = sample['conversations'][0]
        if first_conv.get('from') != 'human':
            return False
        
        # valueフィールドが存在するかチェック
        if 'value' not in first_conv:
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description="第1段階学習用データセット作成")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="/groups/gag51404/ide/PointLLM/data/anno_data/PointLLM_brief_description_660K_filtered.json",
        help="入力データセットファイル"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="/groups/gag51404/ide/PointLLM/data/anno_data/brief_description_stage1_multi_pc.json",
        help="出力データセットファイル"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="ランダムシード"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("第1段階学習用データセット作成ツール")
    print("=" * 60)
    print(f"入力: {args.input_file}")
    print(f"出力: {args.output_file}")
    print(f"シード: {args.seed}")
    
    try:
        # データセット作成器を初期化
        creator = Stage1DatasetCreator(
            input_file=args.input_file,
            output_file=args.output_file,
            seed=args.seed
        )
        
        # データセット作成
        new_dataset = creator.create_dataset()
        
        # データセット保存
        creator.save_dataset(new_dataset)
        
        print("\n" + "=" * 60)
        print("第1段階学習用データセット作成完了！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 
import os
import json
import torch
import numpy as np
import re

import copy
import transformers
from torch.utils.data import Dataset

from .utils import *


def make_object_point_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for Joint3Ddataset with text and point cloud data."""
    """Initialize datasets."""

    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)
    if data_args.split_train_val:
        print("Loading training datasets.")
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            tokenizer=tokenizer,
            use_color=data_args.use_color,
            data_args=data_args
        )
        print("Done!")
        if data_args.data_debug_num > 0:
            print('Debug mode, using training set as val set.')
            val_dataset = train_dataset
        else:
            # * make a val dataset
            print("Loading validation datasets.")
            val_dataset = ObjectPointCloudDataset(
                split='val', # * load train split
                data_path=data_args.data_path,
                anno_path=data_args.anno_path,
                pointnum=data_args.pointnum,
                conversation_types=data_args.conversation_types,
                tokenizer=tokenizer,
                use_color=data_args.use_color,
                data_args=data_args
            )
        return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    else:
        # * use all data as training data
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            use_color=data_args.use_color,
            tokenizer=tokenizer,
            data_args=data_args
        )
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class ObjectPointCloudDataset(Dataset):
    """Dataset utilities for objaverse."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 tokenizer=None,
                 pointnum=8192,
                 split='train',
                 conversation_types=None, # * default is simple_des, used for stage1 pre-train
                 use_color=True,
                 data_args=None):

        """
        split: only considered when data_args.split_train_val is True.
        conversation_types: tuple, used to filter the data, default is ('simple_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        super(ObjectPointCloudDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.data_path = data_path
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.split = split 
        if conversation_types is None:
            self.conversation_types = ("simple_description",)
        else:
            self.conversation_types = conversation_types

        self.data_args = data_args
        self.normalize_pc = True
        self.use_color = use_color

        self.pointnum = pointnum
        self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None
        self.point_indicator = '<point>'

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)
        
        # * print the conversations_type
        print(f"Using conversation_type: {self.conversation_types}") 
        # * print before filtering
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")

        # * iterate the list and filter
        # * these two ids have corrupted colored point files, so filter them when use_color is True
        filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []

        # Iterate the list, filter those "conversation_type" not in self.conversation_types
        def should_filter_data(data):
            # 会話タイプのフィルタリング
            if data.get('conversation_type', 'simple_description') not in self.conversation_types:
                return False
            
            # オブジェクトIDのフィルタリング（単一・複数両対応）
            if 'object_ids' in data:  # 複数点群の場合
                # object_idsのいずれかがfilter_idsに含まれていたら除外
                return not any(obj_id in filter_ids for obj_id in data['object_ids'])
            elif 'object_id' in data:  # 単一点群の場合
                return data['object_id'] not in filter_ids
            else:
                return False  # object_idもobject_idsもない場合は除外
        
        self.list_data_dict = [
            data for data in self.list_data_dict 
            if should_filter_data(data)
        ]

        # * print after filtering
        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")
        # * print the size of different conversation_type
        for conversation_type in self.conversation_types:
            print(f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])}")

        if self.data_args is not None and self.data_args.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[:self.data_args.data_debug_num]
            # * print all the scan_id in debug mode, not using for loop
            debug_ids = []
            for data in self.list_data_dict:
                if 'object_ids' in data:
                    debug_ids.extend(data['object_ids'])
                elif 'object_id' in data:
                    debug_ids.append(data['object_id'])
            print('Debug mode, using: ' + ' '.join(debug_ids))
        elif self.data_args is not None and self.data_args.split_train_val:
            # * split train and val with 9:1 ratios
            if self.split == 'train':
                self.list_data_dict = self.list_data_dict[:int(self.data_args.split_ratio * len(self.list_data_dict))]
                print(f"Train set size: {len(self.list_data_dict)}")
            else:
                self.list_data_dict = self.list_data_dict[int(self.data_args.split_ratio * len(self.list_data_dict)):]
                print(f"Val set size: {len(self.list_data_dict)}")

    def _load_point_cloud(self, object_id, type='objaverse'):
        # Shape Mating データセット用：フルパスが渡された場合は直接読み込み
        if object_id.endswith('.npy') and os.path.exists(object_id):
            point_cloud = np.load(object_id)
            
            # Shape Matingデータは既存のPointBERTモデル（6チャンネル学習済み）との互換性のため
            # use_colorの設定に関わらず常に6チャンネルにパディング
            if point_cloud.shape[1] == 3:
                # 3チャンネル（XYZ）の場合、ダミーRGB値（0.5, 0.5, 0.5）を追加
                num_points = point_cloud.shape[0]
                dummy_rgb = np.full((num_points, 3), 0.5, dtype=point_cloud.dtype)
                point_cloud = np.concatenate([point_cloud, dummy_rgb], axis=1)
            elif point_cloud.shape[1] > 6:
                # 6チャンネル以上の場合、最初の6チャンネルのみを使用
                point_cloud = point_cloud[:, :6]
                
            return point_cloud
        
        # 従来のObjaverseデータセット用
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id) 

    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, f"{self.pointnum}_npy", filename))

        if not self.use_color:
            point_cloud = point_cloud[:, :3]

        return point_cloud

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc
    
    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        
        # ★★★ ここからロジックを変更・追加 ★★★
        
        # プレースホルダーが含まれているかどうかの判定
        # <point> または <pc_1> などが含まれているかチェック
        contains_pc_placeholder = self.point_indicator in sources[0]['conversations'][0]['value'] or \
                                  '<pc_' in sources[0]['conversations'][0]['value']

        if contains_pc_placeholder:
            all_point_clouds = []
            # --- 複数点群の読み込み処理を追加 ---
            if 'object_ids' in sources[0]: # 今回作成した新しいデータ形式
                object_ids = sources[0]['object_ids']
                for obj_id in object_ids:
                    pc = self._load_point_cloud(obj_id)
                    if self.normalize_pc:
                        pc = self.pc_norm(pc)
                    all_point_clouds.append(torch.from_numpy(pc.astype(np.float32)))
                # 複数の点群テンソルをリストとして保持
                point_cloud = all_point_clouds

            # --- 従来の単一点群の処理はそのまま残す ---
            elif 'object_id' in sources[0]: # 従来のデータ形式
                object_id = sources[0]['object_id']
                pc = self._load_point_cloud(object_id)
                if self.normalize_pc:
                    pc = self.pc_norm(pc)
                point_cloud = torch.from_numpy(pc.astype(np.float32))
            else:
                raise ValueError("Sample is missing 'object_id' or 'object_ids' key.")

            if self.tokenizer is None:
                data_dict = dict(
                    point_clouds=point_cloud,
                    object_ids=sources[0].get('object_ids', sources[0].get('object_id'))
                )
                return data_dict

        # --- テキストの前処理部分 ---
        sources = copy.deepcopy([self.list_data_dict[index]["conversations"]])

        # ここから修正
        if contains_pc_placeholder:
            # モデルが内部で使う<point_patch>トークン列を定義
            point_token_len = self.point_backbone_config['point_token_len']
            replace_token = self.point_backbone_config['default_point_patch_token'] * point_token_len
            if self.point_backbone_config['mm_use_point_start_end']:
                replace_token = self.point_backbone_config['default_point_start_token'] + replace_token + self.point_backbone_config['default_point_end_token']

            # すべての識別子（<pc_1>, <pc_2>, ...）を動的に検出して置換する
            conversation_text = sources[0][0]['value']
            # <pc_数字>パターンを検出
            pc_placeholders = re.findall(r'<pc_\d+>', conversation_text)
            for placeholder in set(pc_placeholders):  # 重複を除去
                conversation_text = conversation_text.replace(placeholder, replace_token)
            sources[0][0]['value'] = conversation_text
            
            # 従来の<point>プレースホルダーも処理
            if self.point_indicator in sources[0][0]['value']:
                sources[0][0]['value'] = sources[0][0]['value'].replace(self.point_indicator, replace_token)
        # ここまで修正 

        data_dict = preprocess_v1(
            sources,
            self.tokenizer)

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # point exist in the data
        if contains_pc_placeholder:
            # ★ 修正したpoint_cloud変数を格納する
            data_dict['point_clouds'] = point_cloud

        return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/objaverse_data", type=str,
                        help="Path to the data directory.")
    parser.add_argument("--anno_path", default=None, type=str, required=True,
                        help="Path to the annotation file.")
    parser.add_argument("--split", default='train', type=str, 
                        help="Whether to use the train or validation dataset.")
    parser.add_argument("--pointnum", default=8192, type=int,
                        help="Number of points in the point cloud.")
    parser.add_argument("--data_debug_num", default=0, type=int,
                        help="Number of data to debug with.")
    parser.add_argument("--split_train_val", default=False, type=bool,
                        help="Whether to split the dataset into training and validation.")
    parser.add_argument("--split_ratio", default=0.9, type=float,
                        help="The ratio of training to validation data.")
    parser.add_argument("--tokenizer_path", default=None, type=str, required=True,
                        help="Path to the tokenizer config file.")
    
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    args.point_backbone_config = None

    # Initialize dataset
    dataset = ObjectPointCloudDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        pointnum=args.pointnum,
        split=args.split,
        tokenizer=tokenizer,
        data_args=args
    )

    # Example usage
    print(f'Dataset length: {len(dataset)}')


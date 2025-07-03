#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_scene_qa.py
--------------------------------------------------
• 入力
    – --scene_json      3D-Front QA 定義 JSON（質問・GT を含む）
    – --data_root       /path/to/3D-Front/<scene>/<room>.ply|.pcd
• 出力
    – <scene_json>_PointLLM_outputs.json
      {
        "<scene_id>@<room_id>": {
          "questions": [...],
          "model_answers": [...]
        },
        ...
      }
• GPT-4 等による後段評価は行わない。
--------------------------------------------------
"""

import argparse, json, os, glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d                        # Point cloud loader
from transformers import AutoTokenizer
import numpy as np

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import PointLLMLlamaForCausalLM             # ★ PointLLM 本体
from pointllm.model.utils import KeywordsStoppingCriteria        # 生成停止用

# ------------------------- データセット ------------------------- #
class SceneQADataset(Dataset):
    def __init__(self, scene_json, data_root, pointnum=8192, use_color=True):
        self.entries = []
        with open(scene_json) as fp:
            raw = json.load(fp)

        for object_id, meta in raw.items():
            scene_id = meta["scene_id"]
            room_id  = meta["room_id"]
            # 3D-Front の点群ファイルを探索（.ply or .pcd）
            base = os.path.join(data_root, scene_id, room_id)
            pc_file = None
            # _fps.ply で終わるファイルを探す
            fps_file = os.path.join(base, f"{room_id}_fps.ply")
            if os.path.exists(fps_file):
                pc_file = fps_file
            # もし見つからなければ従来の探索方法をフォールバックとして使用
            if pc_file is None:
                for ext in (".ply", ".pcd"):
                    cand = base + ext
                    if os.path.exists(cand):
                        pc_file = cand; break
            if pc_file is None:
                raise FileNotFoundError(f"Point cloud not found for {object_id}")
            self.entries.append({
                "object_id" : object_id,
                "pc_file"   : pc_file,
                "questions" : meta["questions"]
            })
        self.pointnum   = pointnum
        self.use_color  = use_color

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        pcd = o3d.io.read_point_cloud(e["pc_file"])
        
        # 点群座標を取得
        points_xyz = np.asarray(pcd.points)
        
        # 点の数が足りない場合はアサート
        assert len(points_xyz) >= self.pointnum, f"Point cloud {e['pc_file']} has {len(points_xyz)} points, but {self.pointnum} are required."
        
        # 色情報が必要なので存在確認
        assert hasattr(pcd, 'colors') and len(pcd.colors) > 0, f"Color information not found in {e['pc_file']} but it is required."
        
        # 色情報を取得
        points_rgb = np.asarray(pcd.colors)
        assert len(points_rgb) >= self.pointnum, f"Point cloud {e['pc_file']} has {len(points_rgb)} color values, but {self.pointnum} are required."
        
        # 必要な点数だけ取得
        points_xyz = points_xyz[:self.pointnum]
        points_rgb = points_rgb[:self.pointnum]
        
        # XYZとRGB情報を結合 (必ず6チャンネルになる)
        features = np.concatenate([points_xyz, points_rgb], axis=1)  # (N, 6)
            
        # Numpy配列からTorchテンソルに変換
        pts = torch.from_numpy(features).float()
        
        return {
            "object_id" : e["object_id"],
            "points"    : pts,
            "questions" : e["questions"]
        }

# ------------------------- PointLLM 初期化 ---------------------- #
def init_model(model_name):
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        use_cache=True
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    conv = conv_templates["vicuna_v1_1"].copy()
    # 3D トークン関連パラメータ
    pc_cfg = model.get_model().point_backbone_config
    tokens = (pc_cfg['default_point_patch_token'] * pc_cfg['point_token_len'])
    if pc_cfg['mm_use_point_start_end']:
        tokens = (pc_cfg['default_point_start_token'] + tokens +
                  pc_cfg['default_point_end_token'])
    return model, tokenizer, conv, tokens

# ------------------------- 生成ループ ---------------------------- #
@torch.no_grad()
def qa_inference(model, tokenizer, conv, point_tokens,
                 dataloader, temperature=0.0):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    outputs = {}
    model.eval()

    for batch in tqdm(dataloader, desc="Inference"):
        obj_id  = batch["object_id"][0]           # batch_size=1
        pc      = batch["points"]
        
        # デバッグ用に形状を出力
        print(f"Points tensor shape: {pc.shape}")
        
        # PointLLMが期待する入力形状：[バッチサイズ, 点の数, 次元数]
        # 現在のpcの形状は[点の数, 次元数]なので、バッチ次元を追加
        if len(pc.shape) == 2:
            pc = pc.unsqueeze(0)  # [1, 点の数, 次元数]
            print(f"After unsqueeze, shape: {pc.shape}")
        
        # 入力をモデルと同じbfloat16型に変換
        pc = pc.to(dtype=torch.bfloat16).cuda()
        qs_list = batch["questions"][0]

        model_answers = []
        for qs in qs_list:
            conv_i = conv.copy()
            conv_i.append_message(conv_i.roles[0],
                                  point_tokens + "\n" + qs.strip())
            conv_i.append_message(conv_i.roles[1], None)

            input_ids = tokenizer(
                conv_i.get_prompt(),
                return_tensors="pt"
            ).input_ids.cuda()
            
            # 各プロンプトごとに新しいstopperを初期化
            stopper = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            out_ids = model.generate(
                input_ids,
                point_clouds=pc,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                max_length=2048,
                stopping_criteria=[stopper]
            )
            reply = tokenizer.decode(out_ids[0, input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
            model_answers.append(reply)

        outputs[obj_id] = {
            "questions"     : qs_list,
            "model_answers" : model_answers
        }
    return outputs

# ------------------------- メイン ------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", required=False, default=None,
                        help="QA JSON (scene-level)")
    parser.add_argument("--data_root",  
                        default="/groups/gag51404/ide/PointLLM/data/3d-grand-data/data/sample/3D-Front",
                        help="Root dir of 3D-Front point clouds")
    parser.add_argument("--model_name", default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--pointnum",   type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--use_color",  action="store_true")
    parser.add_argument("--temperature",type=float, default=0.0)
    parser.add_argument("--out_dir",    default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # scene_json が指定されていない場合のデフォルト処理
    if args.scene_json is None:
        # data_root から _fps.ply ファイルを再帰的に探索
        fps_files = []
        for root, dirs, files in os.walk(args.data_root):
            for file in files:
                if file.endswith("_fps.ply"):
                    fps_files.append(os.path.join(root, file))
        
        # 各ファイルに対応する擬似的な scene_json データを作成
        scene_data = {}
        for pc_path in fps_files:
            # パスからシーンIDとルームIDを抽出 
            # 例: /path/to/scene_id/room_id/room_id_fps.ply
            path_parts = pc_path.split(os.sep)
            scene_id = path_parts[-3]  # 3つ上がシーンID
            room_id = path_parts[-2]   # 2つ上がルームID
            object_id = f"{scene_id}@{room_id}"
            
            scene_data[object_id] = {
                "scene_id": scene_id,
                "room_id": room_id,
                "questions": ["What is this?"]  # デフォルトプロンプト
            }
        
        # 一時的なJSONファイルに保存
        temp_json_path = os.path.join(args.out_dir, "temp_scene_data.json")
        with open(temp_json_path, "w") as fp:
            json.dump(scene_data, fp)
        
        # 一時JSONファイルを使用
        args.scene_json = temp_json_path
        out_file = os.path.join(args.out_dir, "default_question_results.json")
    else:
        out_file = os.path.join(
            args.out_dir,
            os.path.splitext(os.path.basename(args.scene_json))[0] +
            "_PointLLM_outputs.json"
        )

    # ---------- Load dataset & model ---------- #
    dataset = SceneQADataset(args.scene_json, args.data_root,
                           args.pointnum, args.use_color)
    dataloader= DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    model, tokenizer, conv, point_tokens = init_model(args.model_name)

    # ---------- Inference ---------- #
    results = qa_inference(model, tokenizer, conv, point_tokens,
                           dataloader, args.temperature)

    # ---------- Save ---------- #
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"[✓] Saved outputs to {out_file}")

if __name__ == "__main__":
    main()

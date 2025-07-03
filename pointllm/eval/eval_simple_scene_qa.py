#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_simple_qa.py
--------------------------------------------------
指定したディレクトリ下にある点群ファイルに対して
特定のプロンプトを実行するシンプルなスクリプト

使用例:
python eval_simple_qa.py \
  --data_dir /path/to/point_clouds \
  --prompt "この点群は何ですか？" \
  --out_file results.json
--------------------------------------------------
"""

import argparse, json, os, glob, tempfile
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from transformers import AutoTokenizer
import numpy as np

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.utils import KeywordsStoppingCriteria

# ------------------------- FPSサンプリング関数 ------------------------- #
def fps(points, n_samples):
    """
    Farthest Point Sampling アルゴリズム
    points: (N, 3) または (N, 6) の点群
    n_samples: サンプリング後の点数
    """
    n_points = points.shape[0]
    if n_points <= n_samples:
        return points  # サンプル数が元の点数以上なら全点を返す
    
    # 最初の点をランダムに選択
    indices = np.zeros(n_samples, dtype=np.int32)
    indices[0] = np.random.randint(n_points)
    
    # 各点から最も近い選択済み点までの距離
    dists = np.ones(n_points) * 1e10
    
    # FPS メインループ
    for i in range(1, n_samples):
        # 最後に選んだ点
        last_idx = indices[i-1]
        last_pt = points[last_idx, :3]  # 座標部分のみ使用
        
        # 各点と最後に選んだ点との距離を計算
        new_dists = np.sum((points[:, :3] - last_pt)**2, axis=1)
        dists = np.minimum(dists, new_dists)
        
        # 最も遠い点を次のサンプルとして選択
        indices[i] = np.argmax(dists)
    
    return points[indices]

def process_pointcloud_fps(input_file, output_file=None, n_samples=8192):
    """点群ファイルを読み込み、FPSでサンプリングして返す。output_fileが指定されていれば保存も行う"""
    # 点群を読み込む
    pcd = o3d.io.read_point_cloud(input_file)
    
    # 点とカラー情報を取得
    points = np.asarray(pcd.points)
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # 点と色を結合
        point_data = np.hstack((points, colors))
        # FPSでサンプリング
        sampled_data = fps(point_data, n_samples)
        # 点と色を分離
        sampled_points = sampled_data[:, :3]
        sampled_colors = sampled_data[:, 3:]
        
        # 新しい点群を作成
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        new_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)
    else:
        # 座標のみでFPSサンプリング
        sampled_points = fps(points, n_samples)
        
        # 新しい点群を作成
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    # 出力ファイルが指定されていれば保存
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        o3d.io.write_point_cloud(output_file, new_pcd)
    
    return new_pcd

def find_fps_files(data_dir, file_extensions):
    """_fpsで終わるファイルを探す"""
    fps_files = []
    all_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                full_path = os.path.join(root, file)
                all_files.append(full_path)
                # ファイル名から拡張子を除いた部分が_fpsで終わるかチェック
                filename, ext = os.path.splitext(file)
                if filename.endswith('_fps'):
                    fps_files.append(full_path)
    
    return fps_files, all_files

# ------------------------- データセット ------------------------- #
class SimplePointCloudDataset(Dataset):
    def __init__(self, data_dir, pointnum=8192, file_extensions=None, use_fps=True):
        """
        data_dir: 点群ファイルを含むディレクトリ
        pointnum: 使用する点の数
        file_extensions: 対象とするファイル拡張子のリスト（デフォルト: ['.ply', '.pcd']）
        use_fps: True の場合、FPSサンプリングを使用（必要な場合）
        """
        if file_extensions is None:
            file_extensions = ['.ply', '.pcd']
        
        self.pointnum = pointnum
        self.pc_files = []
        self.tmp_files = []  # 一時ファイルのリスト（後で削除する）
        
        # _fps で終わるファイルを探す
        fps_files, all_files = find_fps_files(data_dir, file_extensions)
        
        if fps_files:
            print(f"Found {len(fps_files)} _fps files. Using only these files.")
            self.pc_files = fps_files
        else:
            print(f"No _fps files found. Using all {len(all_files)} files with FPS sampling.")
            self.pc_files = all_files
            
            # FPSサンプリングが必要かつ要求されている場合
            if use_fps:
                # 一時ディレクトリを作成
                self.tmp_dir = tempfile.mkdtemp(prefix="pointllm_fps_")
                print(f"Created temporary directory for FPS sampled files: {self.tmp_dir}")
                
                # 全ファイルに対してFPSサンプリングを実行
                new_pc_files = []
                for i, pc_file in enumerate(tqdm(self.pc_files, desc="FPS Sampling")):
                    filename = os.path.basename(pc_file)
                    base, ext = os.path.splitext(filename)
                    tmp_file = os.path.join(self.tmp_dir, f"{base}_fps{ext}")
                    self.tmp_files.append(tmp_file)
                    
                    try:
                        process_pointcloud_fps(pc_file, tmp_file, pointnum)
                        new_pc_files.append(tmp_file)
                    except Exception as e:
                        print(f"Error during FPS sampling of {pc_file}: {e}")
                        # エラーが発生した場合は元のファイルを使用
                        new_pc_files.append(pc_file)
                
                self.pc_files = new_pc_files
        
        if not self.pc_files:
            raise FileNotFoundError(f"No point cloud files found in {data_dir}")
        
        print(f"Using {len(self.pc_files)} point cloud files for inference")

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, idx):
        pc_file = self.pc_files[idx]
        
        try:
            pcd = o3d.io.read_point_cloud(pc_file)
            
            # 点群座標を取得
            points_xyz = np.asarray(pcd.points)
            
            if len(points_xyz) == 0:
                raise ValueError(f"Empty point cloud: {pc_file}")
            
            # 点の数が足りない場合は繰り返しサンプリング
            if len(points_xyz) < self.pointnum:
                # 不足分を繰り返しで補完
                repeat_times = (self.pointnum // len(points_xyz)) + 1
                points_xyz = np.tile(points_xyz, (repeat_times, 1))[:self.pointnum]
            else:
                points_xyz = points_xyz[:self.pointnum]
            
            # 色情報の処理
            if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
                points_rgb = np.asarray(pcd.colors)
                if len(points_rgb) < self.pointnum:
                    repeat_times = (self.pointnum // len(points_rgb)) + 1
                    points_rgb = np.tile(points_rgb, (repeat_times, 1))[:self.pointnum]
                else:
                    points_rgb = points_rgb[:self.pointnum]
                
                # XYZ + RGB (6チャンネル)
                features = np.concatenate([points_xyz, points_rgb], axis=1)
            else:
                # RGB情報がない場合、デフォルト色(白)を追加
                default_rgb = np.ones((self.pointnum, 3)) * 0.5  # グレー
                features = np.concatenate([points_xyz, default_rgb], axis=1)
            
            pts = torch.from_numpy(features).float()
            
            return {
                "file_path": pc_file,
                "file_name": os.path.basename(pc_file),
                "points": pts
            }
            
        except Exception as e:
            print(f"Error loading {pc_file}: {e}")
            # エラーの場合、ダミーデータを返す
            dummy_points = np.random.rand(self.pointnum, 3)
            dummy_rgb = np.ones((self.pointnum, 3)) * 0.5
            dummy_features = np.concatenate([dummy_points, dummy_rgb], axis=1)
            pts = torch.from_numpy(dummy_features).float()
            
            return {
                "file_path": pc_file,
                "file_name": os.path.basename(pc_file),
                "points": pts
            }

    def __del__(self):
        # 一時ファイルを削除
        for tmp_file in self.tmp_files:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass
        
        # 一時ディレクトリを削除（存在する場合）
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            try:
                import shutil
                shutil.rmtree(self.tmp_dir)
                print(f"Removed temporary directory: {self.tmp_dir}")
            except:
                pass

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

# ------------------------- 推論実行 ---------------------------- #
@torch.no_grad()
def run_inference(model, tokenizer, conv, point_tokens,
                  dataloader, prompt, temperature=0.0):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    results = []
    model.eval()

    for batch in tqdm(dataloader, desc="Processing point clouds"):
        file_path = batch["file_path"][0]
        file_name = batch["file_name"][0]
        pc = batch["points"]
        
        # バッチ次元を追加
        if len(pc.shape) == 2:
            pc = pc.unsqueeze(0)
        
        # GPU・型変換
        pc = pc.to(dtype=torch.bfloat16).cuda()
        
        # 会話テンプレートを準備
        conv_i = conv.copy()
        conv_i.append_message(conv_i.roles[0], point_tokens + "\n" + prompt.strip())
        conv_i.append_message(conv_i.roles[1], None)

        # トークン化
        input_ids = tokenizer(
            conv_i.get_prompt(),
            return_tensors="pt"
        ).input_ids.cuda()
        
        # 停止条件
        stopper = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # 生成実行
        try:
            out_ids = model.generate(
                input_ids,
                point_clouds=pc,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                max_length=2048,
                stopping_criteria=[stopper]
            )
            
            reply = tokenizer.decode(
                out_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
        except Exception as e:
            print(f"Error during inference for {file_name}: {e}")
            reply = f"Error: {str(e)}"
        
        results.append({
            "file_path": file_path,
            "file_name": file_name,
            "prompt": prompt,
            "answer": reply
        })
    
    return results

# ------------------------- メイン ------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Simple Point Cloud QA with PointLLM")
    parser.add_argument("--data_dir", default="/groups/gag51404/ide/PointLLM/data/qiu-san-data",
                        help="Directory containing point cloud files")
    parser.add_argument("--prompt", default="What is this?",
                        help="Question/prompt to ask about each point cloud")
    parser.add_argument("--out_file", default="simple_scene_qa_results.json",
                        help="Output JSON file")
    parser.add_argument("--model_name", default="RunsenXu/PointLLM_7B_v1.2",
                        help="PointLLM model name")
    parser.add_argument("--pointnum", type=int, default=8192,
                        help="Number of points to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature")
    parser.add_argument("--file_extensions", nargs="+", 
                        default=[".ply", ".pcd"],
                        help="Point cloud file extensions to process")
    parser.add_argument("--no_fps", action="store_true",
                        help="Disable FPS sampling for files without _fps suffix")
    
    args = parser.parse_args()
    
    # データセット・データローダー
    dataset = SimplePointCloudDataset(
        args.data_dir, 
        args.pointnum,
        args.file_extensions,
        use_fps=not args.no_fps
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # モデル初期化
    print("Loading PointLLM model...")
    model, tokenizer, conv, point_tokens = init_model(args.model_name)
    print("Model loaded successfully!")
    
    # 推論実行
    print(f"Running inference with prompt: '{args.prompt}'")
    results = run_inference(
        model, tokenizer, conv, point_tokens,
        dataloader, args.prompt, args.temperature
    )
    
    # 結果保存
    with open(args.out_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.out_file}")
    print(f"Processed {len(results)} point cloud files")

if __name__ == "__main__":
    main() 
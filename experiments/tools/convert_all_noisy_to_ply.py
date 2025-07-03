import os
import pickle
import numpy as np
import argparse
import glob
from pathlib import Path
from tqdm import tqdm

def save_ply(points, filename, with_normal=True):
    """点群データをPLYファイルとして保存"""
    num_points = len(points)
    
    with open(filename, 'w') as f:
        # ヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if with_normal and points.shape[1] >= 6:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        
        f.write("end_header\n")
        
        # データ部分
        if with_normal and points.shape[1] >= 6:
            for i in range(num_points):
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {points[i, 3]} {points[i, 4]} {points[i, 5]}\n")
        else:
            for i in range(num_points):
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")

def process_dat_file(dat_file, output_base_dir, categories, variant_name=None):
    """単一のdatファイルを処理してPLYに変換"""
    print(f'処理中: {dat_file}...')
    
    # 出力ディレクトリの作成
    if variant_name:
        output_dir = os.path.join(output_base_dir, variant_name)
    else:
        output_dir = output_base_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # datファイルを読み込む
    try:
        with open(dat_file, 'rb') as f:
            list_of_points, list_of_labels = pickle.load(f)
        print(f"  データ総数: {len(list_of_points)}")
    except Exception as e:
        print(f"  エラー: {dat_file}の読み込みに失敗しました - {e}")
        return 0
    
    # クラスごとの処理数をカウントするための辞書
    processed_count = {}
    
    # 各点群データをPLYファイルとして保存
    for i, (points, label) in enumerate(zip(list_of_points, list_of_labels)):
        # カテゴリー名またはラベルインデックスを取得
        if categories:
            category_name = categories[int(label)]
        else:
            category_name = f"class_{int(label)}"
        
        # クラスディレクトリを作成
        class_dir = os.path.join(output_dir, category_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # インデックスを初期化
        if category_name not in processed_count:
            processed_count[category_name] = 0
        
        # ファイル名を作成
        ply_filename = os.path.join(class_dir, f"{category_name}_{processed_count[category_name]:04d}.ply")
        
        # PLYファイルとして保存
        save_ply(points, ply_filename, with_normal=(points.shape[1] >= 6))
        processed_count[category_name] += 1
    
    total_processed = sum(processed_count.values())
    print(f"  変換完了: {total_processed}個のPLYファイルを保存しました")
    return total_processed

# メイン処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ModelNetのノイズ付きデータを一括でPLYに変換します")
    parser.add_argument("--noisy_root", type=str, 
                        default="/home/minesawa/3d/PointLLM/data/modelnet40_data/ModelNet40_noisy",
                        help="ノイズ付きデータのルートディレクトリ")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/minesawa/3d/PointLLM/data/modelnet40_data/ModelNet40_noisy_ply",
                        help="PLYファイルの出力先ディレクトリ")
    parser.add_argument("--category_file", type=str,
                        default="/home/minesawa/3d/PointLLM/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt",
                        help="カテゴリー名のファイル")
    
    args = parser.parse_args()
    
    # カテゴリー名のリストを読み込む
    if os.path.exists(args.category_file):
        categories = [line.rstrip() for line in open(args.category_file)]
        print(f"カテゴリーファイルから{len(categories)}個のクラスを読み込みました")
    else:
        print(f"Warning: カテゴリーファイル {args.category_file} が見つかりません。代わりにラベルインデックスを使用します。")
        categories = None
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # すべてのバリアントディレクトリを取得
    noisy_root = Path(args.noisy_root)
    variant_dirs = [d for d in noisy_root.iterdir() if d.is_dir()]
    
    print(f"{len(variant_dirs)}個のノイズバリアントが見つかりました")
    
    # 全ファイルの処理
    total_files_converted = 0
    
    # 各バリアントを処理
    for variant_dir in tqdm(variant_dirs, desc="バリアント処理"):
        variant_name = variant_dir.name
        print(f"\n処理中のバリアント: {variant_name}")
        
        # バリアント内のすべてのdatファイルを検索
        dat_files = list(variant_dir.glob("*.dat"))
        
        for dat_file in dat_files:
            # 各datファイルをPLYに変換
            files_converted = process_dat_file(
                dat_file, 
                args.output_dir, 
                categories, 
                variant_name
            )
            total_files_converted += files_converted
    
    print(f"\n全処理完了: 合計{total_files_converted}個のPLYファイルを生成しました")
    print(f"出力先: {args.output_dir}")
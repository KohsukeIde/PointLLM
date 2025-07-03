#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resample_pointclouds_fps.py
---------------------------
指定されたディレクトリ下の全ての.plyファイルをFPSで再サンプリングするスクリプト
"""

import os
import glob
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

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

def process_pointcloud(input_file, output_file, n_samples, use_color=True):
    """点群ファイルを読み込み、FPSでサンプリングして保存する"""
    # 点群を読み込む
    pcd = o3d.io.read_point_cloud(input_file)
    
    # 点とカラー情報を取得
    points = np.asarray(pcd.points)
    
    if use_color and pcd.has_colors():
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
        points_only = points
        # サンプリングでインデックスを取得
        n_points = points_only.shape[0]
        if n_points <= n_samples:
            indices = np.arange(n_points)
        else:
            indices = np.zeros(n_samples, dtype=np.int32)
            indices[0] = np.random.randint(n_points)
            dists = np.ones(n_points) * 1e10
            for i in range(1, n_samples):
                last_idx = indices[i-1]
                last_pt = points_only[last_idx]
                new_dists = np.sum((points_only - last_pt)**2, axis=1)
                dists = np.minimum(dists, new_dists)
                indices[i] = np.argmax(dists)
        
        # 選択された点
        sampled_points = points[indices]
        
        # 新しい点群を作成
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        if pcd.has_colors():
            # 選択されたインデックスに対応する色情報を取得
            colors = np.asarray(pcd.colors)
            sampled_colors = colors[indices]
            new_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)
    
    # 保存
    o3d.io.write_point_cloud(output_file, new_pcd)
    return new_pcd

def main():
    parser = argparse.ArgumentParser(description='3D-Front点群をFPSで再サンプリングする')
    parser.add_argument('--root_dir', default='/groups/gag51404/ide/PointLLM/data/3d-grand-data/data/sample/3D-Front',
                        help='3D-Frontデータのルートディレクトリ')
    parser.add_argument('--output_dir', default=None,
                        help='出力ディレクトリ（デフォルトは入力ファイルと同じ場所に_fps.plyで保存）')
    parser.add_argument('--num_points', type=int, default=8192,
                        help='サンプリング後の点数 (デフォルト: 8192)')
    parser.add_argument('--use_color', action='store_true',
                        help='カラー情報を使用するかどうか')
    args = parser.parse_args()
    
    # ルートディレクトリ下の全ての.plyファイルを再帰的に探索
    ply_files = glob.glob(os.path.join(args.root_dir, '**', '*.ply'), recursive=True)
    
    print(f"Found {len(ply_files)} .ply files")
    
    for ply_file in tqdm(ply_files, desc="Resampling point clouds"):
        # 出力ファイル名を決定
        if args.output_dir:
            # 相対パスを維持しながら出力ディレクトリに保存
            rel_path = os.path.relpath(ply_file, args.root_dir)
            output_file = os.path.join(args.output_dir, rel_path)
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            # 元のファイル名の拡張子前に_fpsを付ける
            base, ext = os.path.splitext(ply_file)
            output_file = f"{base}_fps{ext}"
        
        # 処理
        try:
            process_pointcloud(ply_file, output_file, args.num_points, args.use_color)
        except Exception as e:
            print(f"Error processing {ply_file}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()
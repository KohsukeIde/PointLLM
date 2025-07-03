import os
import pickle
import numpy as np

# 確認したいdatファイルのパス
dat_file_path = "/home/minesawa/3d/PointLLM/data/modelnet40_data/ModelNet40_noisy/kr0.30_js0.000_sh0/modelnet40_test_8192pts_fps_noisy.dat"

# カテゴリー名のファイルパス（modelnet.pyから参照）
catfile = "modelnet_config/modelnet40_shape_names_modified.txt"
categories = [line.rstrip() for line in open(catfile)] if os.path.exists(catfile) else None

# datファイルを読み込む
print(f'Loading data from {dat_file_path}...')
with open(dat_file_path, 'rb') as f:
    list_of_points, list_of_labels = pickle.load(f)

# 基本情報の表示
print(f"データの総数: {len(list_of_points)}")
print(f"ラベルの総数: {len(list_of_labels)}")

# 点群データの形状と型を確認
print(f"\n点群データの形状: {list_of_points[0].shape}")
print(f"点群データの型: {type(list_of_points[0])}")
print(f"点群データの例 (最初の3点):\n{list_of_points[0][:3]}")

# ラベルの確認
print(f"\nラベルの型: {type(list_of_labels[0])}")
print(f"ラベルの範囲: {min(list_of_labels)} から {max(list_of_labels)}")

# ラベルの分布
unique_labels, counts = np.unique(list_of_labels, return_counts=True)
print("\nラベルの分布:")
for label, count in zip(unique_labels, counts):
    category_name = categories[int(label)] if categories else f"カテゴリー {label}"
    print(f"  ラベル {label} ({category_name}): {count}個")

# 点の座標範囲を確認
all_points = np.vstack([points[:, :3] for points in list_of_points])
min_coords = np.min(all_points, axis=0)
max_coords = np.max(all_points, axis=0)
print(f"\n点の座標範囲:")
print(f"  X: {min_coords[0]:.4f} から {max_coords[0]:.4f}")
print(f"  Y: {min_coords[1]:.4f} から {max_coords[1]:.4f}")
print(f"  Z: {min_coords[2]:.4f} から {max_coords[2]:.4f}")

# 法線ベクトルがあれば確認
if list_of_points[0].shape[1] >= 6:
    all_normals = np.vstack([points[:, 3:6] for points in list_of_points])
    min_normals = np.min(all_normals, axis=0)
    max_normals = np.max(all_normals, axis=0)
    print(f"\n法線ベクトルの範囲:")
    print(f"  NX: {min_normals[0]:.4f} から {max_normals[0]:.4f}")
    print(f"  NY: {min_normals[1]:.4f} から {max_normals[1]:.4f}")
    print(f"  NZ: {min_normals[2]:.4f} から {max_normals[2]:.4f}")

# 最初の数個のサンプルを詳細に確認
num_samples_to_check = 5
print(f"\n最初の{num_samples_to_check}個のサンプルの詳細:")
for i in range(min(num_samples_to_check, len(list_of_points))):
    points = list_of_points[i]
    label = list_of_labels[i]
    category_name = categories[int(label)] if categories else f"カテゴリー {label}"
    print(f"サンプル {i}:")
    print(f"  ラベル: {label} ({category_name})")
    print(f"  点数: {points.shape[0]}")
    print(f"  特徴量数: {points.shape[1]}")
    print(f"  点の範囲: X[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}], "
          f"Y[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}], "
          f"Z[{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")
    print()
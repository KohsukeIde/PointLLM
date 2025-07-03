import os
import pickle
import numpy as np

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
    
    print(f"Saved {filename}")

# メイン処理
if __name__ == "__main__":
    # 確認したいdatファイルのパス
    dat_file_path = "/home/minesawa/3d/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
    
    # カテゴリー名のファイルパス
    catfile = "/home/minesawa/3d/PointLLM/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt"
    
    # 出力ディレクトリ
    output_dir = "modelnet40/pointclouds"
    
    # カテゴリー名のリスト
    if os.path.exists(catfile):
        categories = [line.rstrip() for line in open(catfile)]
    else:
        print(f"Warning: カテゴリーファイル {catfile} が見つかりません。代わりにラベルインデックスを使用します。")
        categories = None
    
    # datファイルを読み込む
    print(f'Loading data from {dat_file_path}...')
    with open(dat_file_path, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)
    
    print(f"データの総数: {len(list_of_points)}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # クラスごとの処理数をカウントするための辞書
    processed_count = {}
    skipped_count = {}
    
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
            skipped_count[category_name] = 0
        
        # ファイル名を作成
        ply_filename = os.path.join(class_dir, f"{category_name}_{processed_count[category_name]:04d}.ply")
        
        # ファイルが既に存在する場合はスキップ
        if os.path.exists(ply_filename):
            skipped_count[category_name] += 1
            continue
        
        # PLYファイルとして保存
        save_ply(points, ply_filename, with_normal=True)
        processed_count[category_name] += 1
        
        # 進捗表示
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(list_of_points)} processed")
    
    # 処理結果の表示
    print("\n処理完了:")
    print(f"合計: {len(list_of_points)}個のデータ")
    print(f"クラス数: {len(processed_count)}個のクラス")
    
    # 各クラスの処理数を表示
    print("\nクラスごとの処理数:")
    for category, count in processed_count.items():
        total = count + skipped_count.get(category, 0)
        print(f"  {category}: {count}個処理, {skipped_count.get(category, 0)}個スキップ (合計: {total}個)")
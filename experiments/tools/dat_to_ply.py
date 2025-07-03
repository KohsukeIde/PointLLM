import os
import pickle
import numpy as np
import glob
import re
import argparse
from tqdm import tqdm

def farthest_point_sample(points, npoint):
    """
    ファーザストポイントサンプリング（FPS）を実行
    Input:
        points: [N, D] - 入力点群 (XYZ + 追加特徴量)
        npoint: int - サンプリングする点の数
    Return:
        centroids: [npoint] - 選択された点のインデックス
    """
    N, D = points.shape
    xyz = points[:, :3]  # 座標部分のみ使用
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    
    # 最初の点はランダムに選択
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    # 選択されたインデックスを返す
    return points[centroids]

def pc_normalize(pc):
    """
    点群を単位球に正規化
    Input:
        pc: [N, D] - 入力点群
    Return:
        pc: [N, D] - 正規化された点群
    """
    xyz = pc[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    
    # 最大距離を計算
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    
    # 単位球に正規化
    xyz = xyz / m
    
    # 座標部分を更新
    pc_normalized = pc.copy()
    pc_normalized[:, :3] = xyz
    
    return pc_normalized

def read_ply(filename):
    """PLYファイルから点群データを読み込む（色情報も含む）"""
    try:
        # バイナリ形式のPLYの場合はopen3dを使用
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)
            
            # 色情報があれば読み込む
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)  # すでに0-1の範囲
                return np.concatenate([points, colors], axis=1)
            else:
                # 色情報がない場合は黒色（0,0,0）を割り当て
                colors = np.zeros((points.shape[0], 3))
                return np.concatenate([points, colors], axis=1)
        except:
            pass
        
        # open3dが失敗した場合やインストールされていない場合はテキスト解析
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # ヘッダー情報を解析
        vertex_count = 0
        has_color = False
        header_end = 0
        
        for i, line in enumerate(lines):
            if "element vertex" in line:
                vertex_count = int(line.split()[-1])
            if "property float r" in line or "property uchar r" in line or "property float red" in line:
                has_color = True
            if line.strip() == "end_header":
                header_end = i + 1
                break
        
        # データ部分を読み込む
        points = []
        for i in range(header_end, header_end + vertex_count):
            if i >= len(lines):
                break
                
            values = lines[i].split()
            point = [float(values[0]), float(values[1]), float(values[2])]  # x, y, z
            
            # 法線情報とRGB情報の位置を確認
            has_normal = False
            color_index = 3
            
            # 法線情報があれば位置を調整
            if len(values) >= 6 and not has_color:
                has_normal = True
                point.extend([float(values[3]), float(values[4]), float(values[5])])  # nx, ny, nz
                color_index = 6
            
            # 色情報があれば追加
            if has_color and len(values) > color_index + 2:
                r = float(values[color_index])
                g = float(values[color_index+1])
                b = float(values[color_index+2])
                
                # 値が1より大きい場合は0-255のスケールと判断し、0-1に正規化
                if r > 1.0 or g > 1.0 or b > 1.0:
                    r, g, b = r/255.0, g/255.0, b/255.0
                    
                if has_normal:
                    point = point[:6]  # xyz + normal までを保持
                else:
                    point = point[:3]  # xyz のみ保持
                
                point.extend([r, g, b])  # r, g, b
            elif not has_color:
                # 色情報がない場合は黒色（0,0,0）を割り当て
                if has_normal:
                    point = point[:6]  # xyz + normal までを保持
                else:
                    point = point[:3]  # xyz のみ保持
                point.extend([0.0, 0.0, 0.0])  # 黒色
            
            points.append(point)
        
        return np.array(points, dtype=np.float32)
    
    except Exception as e:
        print(f"PLYファイル読み込みエラー({filename}): {e}")
        return None

def convert_ply_to_dat(input_dir, output_file, category_file=None, sample_points=None):
    """PLYファイルのディレクトリから.datファイルを生成"""
    # カテゴリー名のリスト
    if category_file and os.path.exists(category_file):
        categories = [line.rstrip() for line in open(category_file)]
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    else:
        print("カテゴリーファイルが指定されていないか見つかりません。ディレクトリ名をカテゴリとして使用します。")
        categories = []
        category_to_idx = {}
    
    # 全PLYファイルを検索
    all_ply_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.ply'):
                all_ply_files.append(os.path.join(root, file))
    
    print(f"見つかったPLYファイル: {len(all_ply_files)}個")
    
    list_of_points = []
    list_of_labels = []
    
    # 各ファイルを処理
    for i, ply_file in enumerate(tqdm(all_ply_files, desc="PLYファイル処理中")):
        # ディレクトリ構造からカテゴリを推定（例：床/床_0001.ply → カテゴリは「床」）
        parts = os.path.normpath(ply_file).split(os.sep)
        if len(parts) >= 2:
            category_name = parts[-2]  # ディレクトリ名をカテゴリとして使用
        else:
            # ファイル名からカテゴリを抽出する別の方法
            match = re.search(r'([a-zA-Z_]+)_\d+\.ply', os.path.basename(ply_file))
            if match:
                category_name = match.group(1)
            else:
                print(f"警告: {ply_file} からカテゴリを特定できません。スキップします。")
                continue
        
        # カテゴリインデックスを取得または作成
        if category_name not in category_to_idx:
            category_to_idx[category_name] = len(categories)
            categories.append(category_name)
        
        label = category_to_idx[category_name]
        
        # PLYファイルを読み込み
        try:
            points = read_ply(ply_file)
            if points is None:
                continue
                
            # 必要なら最低6次元になるように（xyz + rgb）
            if points.shape[1] < 6:
                # XYZ座標のみの場合、RGB情報を0として追加
                if points.shape[1] == 3:
                    rgb = np.zeros((points.shape[0], 3))
                    points = np.concatenate([points, rgb], axis=1)
                    
            # 座標（XYZ）と色情報（RGB）のみを保持 - 6列目までにする
            points = points[:, :6]
            
            # サンプリングが指定されていれば実行
            if sample_points is not None and points.shape[0] > sample_points:
                points = farthest_point_sample(points, sample_points)
            elif sample_points is not None and points.shape[0] < sample_points:
                # 点が足りない場合は、既存の点をランダムに複製して増やす
                # （理想的ではないが、データ形式を統一するため）
                idx = np.random.choice(points.shape[0], sample_points - points.shape[0])
                extra_points = points[idx]
                points = np.concatenate([points, extra_points], axis=0)
            
            # 正規化（PointLLM用）
            points = pc_normalize(points)
            
            # データを追加
            list_of_points.append(points)
            list_of_labels.append(label)
                
        except Exception as e:
            print(f"エラー: {ply_file} の処理中に例外が発生しました: {e}")
    
    # .datファイルにデータを保存
    with open(output_file, 'wb') as f:
        pickle.dump([list_of_points, list_of_labels], f)
    
    print(f"\n処理完了:")
    print(f"合計: {len(list_of_points)}個のデータを {output_file} に保存しました")
    print(f"カテゴリー数: {len(categories)}個")
    
    # カテゴリーファイルを保存（もし元々なかった場合）
    if category_file and not os.path.exists(category_file):
        with open(category_file, 'w') as f:
            for category in categories:
                f.write(f"{category}\n")
        print(f"カテゴリーリストを {category_file} に保存しました")
    
    # 各カテゴリの処理数を表示
    print("\nカテゴリごとの処理数:")
    category_counts = {}
    for label in list_of_labels:
        cat_name = categories[label]
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}個")
    
    return list_of_points, list_of_labels, categories

# メイン処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PLYファイルをModelNet形式の.datファイルに変換（FPSサンプリング対応）')
    parser.add_argument('--input_dir', type=str, required=True, help='PLYファイルが含まれるディレクトリ')
    parser.add_argument('--output_file', type=str, required=True, help='出力する.datファイルのパス')
    parser.add_argument('--category_file', type=str, default=None, help='カテゴリリストファイルのパス（存在しない場合は自動生成）')
    parser.add_argument('--sample_points', type=int, default=8192, help='FPSサンプリングする点の数（デフォルト: 8192）')
    parser.add_argument('--no_normalize', action='store_true', help='点群の正規化を行わない場合に指定')
    
    args = parser.parse_args()
    
    # PLY to DAT変換を実行
    list_of_points, list_of_labels, categories = convert_ply_to_dat(
        args.input_dir, 
        args.output_file,
        args.category_file,
        args.sample_points
    )
    
    print(f"\n変換結果:")
    print(f"- 最初の点群形状: {list_of_points[0].shape if list_of_points else 'なし'}")
    print(f"- 点群数: {len(list_of_points)}")
    print(f"- 保存ファイル: {args.output_file}")
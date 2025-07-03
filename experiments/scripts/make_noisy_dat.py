#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_noisy_variants.py  (fixed: always save exactly N points)

- LiDAR 欠損 → Gaussian jitter → shuffle
- その後 **必ず target_npoints(=8192) に整形**
    • 足りない分は重複サンプリングで補充（replace=True）
    • 多い場合はランダムに間引き（replace=False）
"""
import argparse, itertools, pathlib, numpy as np, tqdm, pickle, os

# ----------------------- ノイズ関数 -----------------------
def lidar_dropout(pts: np.ndarray, keep_ratio: float) -> np.ndarray:
    if keep_ratio >= 0.999:
        return pts
    mask = np.random.rand(len(pts)) < keep_ratio
    return pts[mask]

def gaussian_jitter(pts: np.ndarray, jitter_std: float) -> np.ndarray:
    if jitter_std <= 0:
        return pts
    xyz = pts[:, :3]
    scale = (xyz.max(0) - xyz.min(0)).max()      # bbox 対角 ≈ 最大辺
    xyz += np.random.randn(*xyz.shape) * jitter_std * scale
    pts[:, :3] = xyz
    return pts

def shuffle_points(pts: np.ndarray) -> np.ndarray:
    return pts[np.random.permutation(len(pts))]

# ----------------------- リサンプリング -----------------------
def resample_to_fixed(pts: np.ndarray, npoints: int) -> np.ndarray:
    n = len(pts)
    if n == npoints:
        return pts
    idx = np.random.choice(n, npoints, replace=(n < npoints))
    return pts[idx]

# ----------------------- 1 ファイル処理 -----------------------
def process_data(list_of_points, list_of_labels,
                 keep_ratio: float, jitter_std: float, do_shuffle: bool,
                 target_n: int):
    noisy_points = []
    for pts in tqdm.tqdm(list_of_points, desc='  processing', leave=False):
        p = pts.copy()
        p = lidar_dropout(p, keep_ratio)
        p = gaussian_jitter(p, jitter_std)
        if do_shuffle:
            p = shuffle_points(p)
        p = resample_to_fixed(p, target_n)        # ★ここで 8192 点に揃える
        noisy_points.append(p)
    return noisy_points, list_of_labels

# ----------------------- メイン -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root',
                    default="/groups/gag51404/ide/PointLLM/data/modelnet40_data",
                    help='ModelNet データが格納されているディレクトリ')
    ap.add_argument('--src_file',
                    default="modelnet40_test_8192pts_fps.dat",
                    help='ピクル化された元データファイル名')
    ap.add_argument('--dst_root',
                    default="/groups/gag51404/ide/PointLLM/data/ModelNet40_noisy",
                    help='書き出し先 root')
    ap.add_argument('--keep_ratios', default="1.0,0.7,0.5,0.3")
    ap.add_argument('--jitter_stds', default="0,0.001,0.005")
    ap.add_argument('--with_shuffle', choices=['yes', 'no', 'only'], default='yes')
    ap.add_argument('--target_npoints', type=int, default=8192,
                    help='各点群の最終点数 (既定 8192)')
    args = ap.parse_args()

    src_root = pathlib.Path(args.src_root)
    dst_root = pathlib.Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    keep_ratios   = [float(x) for x in args.keep_ratios.split(',')]
    jitter_stds   = [float(x) for x in args.jitter_stds.split(',')]
    shuffle_flags = {'yes': [False, True],
                     'no':  [False],
                     'only':[True]}[args.with_shuffle]

    # 元データ読み込み
    dat_file_path = src_root / args.src_file
    print(f'[INFO] Loading {dat_file_path}')
    with open(dat_file_path, 'rb') as f:
        list_of_points, list_of_labels = pickle.load(f)

    # バリエーション生成
    for kr, js, sh in itertools.product(keep_ratios, jitter_stds, shuffle_flags):
        variant = f"kr{kr:.2f}_js{js:.3f}_sh{int(sh)}"
        print(f"\n[Variant] {variant}")

        noisy_pts, labels = process_data(list_of_points, list_of_labels,
                                         keep_ratio=kr, jitter_std=js, do_shuffle=sh,
                                         target_n=args.target_npoints)

        out_dir = dst_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{os.path.splitext(args.src_file)[0]}_noisy.dat"
        print(f"  → saving to {out_path}")

        with open(out_path, 'wb') as f:
            pickle.dump((noisy_pts, labels), f)

    print("\nAll variants finished!")

if __name__ == '__main__':
    main()

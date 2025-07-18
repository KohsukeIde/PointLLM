#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ood_judge_internvl.py
──────────────────────
• PointLLM が出力した OOD JSON を読み込み
• 各点群を 3 視点レンダリング (matplotlib)
• InternVL-2-40B で「Correct / Incorrect」を判定
• verdict を JSON ＆ CSV に保存し Accuracy を表示
"""

# ───────── Imports ──────────────────────────────────────────────
import argparse, json, math, os, pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ───────── InternVL 設定 ────────────────────────────────────────
MODEL_ID = "OpenGVLab/InternVL2-40B"      # 量子化なら …-AWQ
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ───────── ModelNet-40 クラス一覧 ───────────────────────────────
MODELNET_CLASSES = [
    "airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car","chair",
    "cone","cup","curtain","desk","door","dresser","flower pot","glass box",
    "guitar","keyboard","lamp","laptop","mantel","monitor","night stand",
    "person","piano","plant","radio","range hood","sink","sofa","stairs",
    "stool","table","tent","toilet","tv stand","vase","wardrobe","xbox"
]

# ───────── 画像前処理 (448², ViT-L/14) ───────────────────────────
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_448 = T.Compose([
    T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(), T.Normalize(MEAN, STD)
])

def imgs_to_pixel_values(imgs: List[Image.Image]) -> torch.Tensor:
    return torch.stack([_448(im) for im in imgs]).to(torch.bfloat16).to(DEVICE)

# ───────── 点群を主成分に合わせる関数 ──────────────────────────
def align_principal_axis(pc: np.ndarray) -> np.ndarray:
    """
    点群を主成分座標系 (x = 最長, y = 2 番目, z = 最短) に回転して返す。
    各主成分の符号は "重心が +X 側に伸びる" ようにそろえる。
    """
    cen = pc.mean(0)
    centered = pc - cen
    # 3×3 の固有ベクトル行列 (列 = 主成分)
    eigvec = np.linalg.svd(centered, full_matrices=False)[2].T
    rotated = centered @ eigvec            # PCA 軸へ回転

    # ── 符号そろえ：最大座標が + 側になるよう調整 ──
    for k in range(3):
        if rotated[:, k].max() < -rotated[:, k].min():
            rotated[:, k] *= -1
    return rotated

# ───────── 点群 → PIL.Image (224², 可視化用) ───────────────────
def pc_to_image(pc: np.ndarray, elev: int, azim: int) -> Image.Image:
    """scatter をアップ気味に描画して PIL.Image を返す"""
    # ─ centering & scale ─
    cen  = pc.mean(0)
    span = np.max(np.linalg.norm(pc - cen, axis=1))
    lim  = span * 0.6                # アップ表示に調整

    fig = plt.figure(figsize=(2.4, 2.4), dpi=120)
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev, azim)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               s=3.0, c="k", alpha=0.8, rasterized=True)  # 点を大きく
    ax.set_xlim(cen[0]-lim, cen[0]+lim)
    ax.set_ylim(cen[1]-lim, cen[1]+lim)
    ax.set_zlim(cen[2]-lim, cen[2]+lim)
    ax.set_axis_off(); fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(*fig.canvas.get_width_height()[::-1], 3)
    plt.close(fig)
    return Image.fromarray(img)

# ───────── InternVL で判定 ───────────────────────────────────
def judge_internvl(model, tok, imgs: List[Image.Image],
                   question: str, answer: str) -> str:
    prompt = (
        "<image>\n"                                     # 公式 placeholder
        "You are a strict evaluator. "
        "Look at the images carefully, then read the question and the answer. "
        "If the answer is entirely correct and complete, reply `Correct`; "
        "otherwise reply `Incorrect`.\n\n"
        f"### Question:\n{question}\n\n### Answer:\n{answer}\n\n"
        "### Reply with only one word: Correct or Incorrect."
    )
    pixel_values = imgs_to_pixel_values(imgs)
    cfg = dict(max_new_tokens=4, do_sample=False)
    reply = model.chat(tok, pixel_values, prompt, cfg).strip().lower()
    return "Correct" if reply.startswith("correct") else "Incorrect"

# ───────── Main ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood_json", required=True)
    ap.add_argument("--out_dir", default="judge_results")
    ap.add_argument("--render_dir", default="render_images",
                    help="レンダリング画像の保存先")
    args = ap.parse_args()
    
    pathlib.Path(args.out_dir).mkdir(exist_ok=True)
    pathlib.Path(args.render_dir).mkdir(exist_ok=True)

    with open(args.ood_json) as f: obj = json.load(f)
    prompt_text, records = obj["prompt_text"], obj["results"]

    from pointllm.data import ModelNet
    ds = ModelNet(config_path=None, split="test",
                  subset_nums=-1, use_color=False)

    print("[INFO] loading InternVL-2-40B …")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False)

    # ───────── 角度リスト（6枚の最適視点）─────────
    views = [( 30,  y) for y in (0, 60, 120)] + \
            [(-30,  y) for y in (0, 60, 120)]     # pitch, yaw の順

    judged = []
    for rec in tqdm(records, desc="Judge"):
        idx = rec["object_id"]
        pc_raw = ds[idx]["point_clouds"].cpu().numpy()
        pc_align = align_principal_axis(pc_raw)          # ★ PCAで整列
        
        # ループ内で画像化と保存
        imgs = []
        for pitch, yaw in views:
            img = pc_to_image(pc_align, pitch, yaw)  # 整列済み点群を使用
            
            # レンダリング画像を保存
            img_path = os.path.join(args.render_dir, f"{idx}_p{pitch}_y{yaw}.png")
            img.save(img_path)
            
            imgs.append(img)

        full_q = (
            f"{prompt_text}\n\n"
            f"The 40 possible classes are:\n{', '.join(MODELNET_CLASSES)}\n"
            f"(The object ID is {idx}.)"
        )
        verdict = judge_internvl(model, tok, imgs, full_q, rec["model_output"])
        rec["internvl_verdict"] = verdict
        judged.append(rec)

    # 保存 & 集計
    stem = pathlib.Path(args.ood_json).stem + "_judged"
    out_json = pathlib.Path(args.out_dir) / f"{stem}.json"
    with open(out_json, "w") as f:
        json.dump({"prompt": prompt_text, "results": judged},
                  f, indent=2, ensure_ascii=False)

    correct = sum(r["internvl_verdict"] == "Correct" for r in judged)
    acc = correct / len(judged)
    print(f"[METRIC] Accuracy: {acc:.3f} ({correct}/{len(judged)})")

    pd.DataFrame(
        [(r["object_id"], r["label_name"],
          r["model_output"], r["internvl_verdict"]) for r in judged],
        columns=["object_id", "label_name",
                 "model_output", "internvl_verdict"]
    ).to_csv(out_json.with_suffix(".csv"), index=False)
    print(f"[INFO] saved → {out_json} (+ .csv)")

if __name__ == "__main__":
    main()





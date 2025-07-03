#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pointllm_ood_generate_and_prepare.py

• Point-LLM で 3D OOD プロンプト (A–G) を実行し JSON を保存
• その JSON を人手評価しやすい CSV に変換 (ヘッダ + 空列付き)

依存:
  - pointllm==<your version>
  - torch, tqdm, transformers, pandas (CSV 書き出しで使用)

使用例:
  python pointllm_ood_generate_and_prepare.py \
      --model_name RunsenXu/PointLLM_7B_v1.2 \
      --prompt_index 3 \
      --generate \
      --prepare_csv
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import ModelNet
from transformers import AutoTokenizer

# -------- 1. OOD PROMPT DEFINITIONS --------------------------------------- #
# 7-type prompt list (indexで選択). 先頭に p_tokens を後付けする部分は
# run 時に埋め込むので、ここでは "質問部" のみを書く。
OOD_PROMPTS: List[str] = [
    # A  Affordance (≤15 words)
    "Use at most 15 words: describe one typical everyday use of this object.",
    # B  Cannot-do list (2 items)
    "List two things this object definitely cannot do. Use the form:\n1. ...\n2. ...",
    # C  3D-POPE style: property existence (Yes/No)
    "Does this object have wheels? Answer \"Yes.\" or \"No.\"",
    # D  Attribute Yes/No
    "Is the main surface smooth? Answer \"Yes.\" or \"No.\"",
    # E  Counting
    # "What is this, and how many legs does this object have?",
    "How many legs does this object have? Reply with a single integer.",
    # F  Short caption (≤12 words)
    "Describe this object in ≤ 12 words.",
    # G  Counter-fact physics Yes/No
    "If dropped from 1 m, will it likely break? Yes/No."
]
# -------------------------------------------------------------------------- #


def init_model(model_name: str):
    """ロード & tokenizer, conv テンプレ初期化"""
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    print(f"[INFO] Model name: {os.path.basename(model_name)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
        use_cache=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv


def load_dataset(split: str, subset_nums: int, use_color: bool):
    print(f"[INFO] Loading ModelNet {split} split.")
    dataset = ModelNet(config_path=None, split=split,
                       subset_nums=subset_nums, use_color=use_color)
    print("[INFO] Dataset ready.")
    return dataset


def get_dataloader(dataset, batch_size, num_workers=4):
    """shuffle=False 固定 (object_id を保持)"""
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


@torch.inference_mode()
def generate_outputs(
    model,
    tokenizer,
    input_ids,
    point_clouds,
    stopping_criteria,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    max_length=2048,
    top_p=0.95,
):
    """B 個同時生成 -> str list"""
    output_ids = model.generate(
        input_ids,
        point_clouds=point_clouds,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        max_length=max_length,
        top_p=top_p,
        stopping_criteria=[stopping_criteria],
    )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True)
    return [o.strip() for o in outputs]


def build_prompt(conv, qs: str, model) -> str:
    """
    点群トークンを会話の先頭に挿入して 1 complete prompt を返す
    """
    point_cfg = model.get_model().point_backbone_config
    pt_len = point_cfg["point_token_len"]
    pt_patch = point_cfg["default_point_patch_token"]
    pt_start = point_cfg["default_point_start_token"]
    pt_end = point_cfg["default_point_end_token"]
    use_start_end = point_cfg["mm_use_point_start_end"]
    
    conv.messages = []

    # --- 質問前にダミー point トークン列を置く ---------------------- #
    if use_start_end:
        prefix = f"{pt_start}{pt_patch*pt_len}{pt_end}\n"
    else:
        prefix = f"{pt_patch*pt_len}\n"

    conv.append_message(conv.roles[0], prefix + qs)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()


def run_generation(
    model,
    tokenizer,
    conv,
    dataloader,
    prompt_index: int,
    output_path: Path,
):
    """1 prompt でデータローダ全体を推論し JSON 保存 -> return dict"""
    qs = OOD_PROMPTS[prompt_index]
    stop_str = conv.sep if conv.sep_style != \
        SeparatorStyle.TWO else conv.sep2

    # 共通入力 (1 行) を事前エンコード
    one_prompt = build_prompt(conv, qs, model)
    one_input = tokenizer([one_prompt], return_tensors="pt").input_ids.cuda()

    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, one_input)

    responses: List[Dict] = []
    for batch in tqdm(dataloader, desc="Infer"):
        point_clouds = batch["point_clouds"].cuda().to(model.dtype)
        labels = batch["labels"]
        label_names = batch["label_names"]
        indices = batch["indice"]

        batch_input_ids = one_input.repeat(point_clouds.size(0), 1)
        outs = generate_outputs(
            model, tokenizer, batch_input_ids, point_clouds, stopping
        )

        for idx, out, lab, lab_name in zip(indices, outs, labels, label_names):
            responses.append(
                {
                    "object_id": int(idx.item()),
                    "ground_truth": int(lab.item()),
                    "label_name": lab_name,
                    "model_output": out,
                }
            )

    results_dict = {
        "prompt_id": prompt_index,
        "prompt_text": qs,
        "results": responses,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results_dict, fp, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved JSON -> {output_path}")

    return results_dict


# --------------------  HUMAN CSV PREPARATION ------------------------------ #
CSV_HEADERS = [
    "object_id",
    "label_name",
    "prompt_id",
    "prompt_text",
    "model_output",
    # ↓ ここから人が入力する列
    "format_ok (1/0)",
    "content_ok (1/0)",
    "comments",
]


def prepare_csv_for_human(json_dict: Dict, csv_path: Path):
    """結果 JSON から評価用空白列付き CSV を生成"""
    rows = []
    for item in json_dict["results"]:
        rows.append(
            [
                item["object_id"],
                item["label_name"],
                json_dict["prompt_id"],
                json_dict["prompt_text"],
                item["model_output"],
                "",  # format_ok
                "",  # content_ok
                "",  # comments
            ]
        )

    df = pd.DataFrame(rows, columns=CSV_HEADERS)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved human-eval CSV -> {csv_path}")


# ------------------------------  MAIN  ------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    # model / dataset
    parser.add_argument("--model_name", type=str,
                        default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--use_color", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_nums", type=int, default=-1)

    # prompt & tasks
    parser.add_argument("--prompt_index", type=int, default=0,
                        choices=list(range(len(OOD_PROMPTS))))
    parser.add_argument("--generate", action="store_true",
                        help="Run inference & save JSON")
    parser.add_argument("--prepare_csv", action="store_true",
                        help="Convert JSON to blank CSV for human rating")

    # misc
    parser.add_argument("--output_root", type=str,
                        default="evaluation/prompt_ood")

    args = parser.parse_args()

    # ---------- paths ----------
    out_dir = Path(args.output_root) / os.path.basename(args.model_name)
    json_path = out_dir / f"ModelNet_OOD_prompt{args.prompt_index}.json"
    csv_path = json_path.with_suffix(".csv")

    # ---------- generation ----------
    if args.generate:
        dataset = load_dataset(args.split, args.subset_nums, args.use_color)
        dataloader = get_dataloader(dataset, args.batch_size, args.num_workers)

        model, tokenizer, conv = init_model(args.model_name)
        run_generation(model, tokenizer, conv, dataloader,
                       args.prompt_index, json_path)

        # CUDA メモリ後片付け
        del model, tokenizer
        torch.cuda.empty_cache()

    # ---------- CSV -----------
    if args.prepare_csv:
        if not json_path.exists():
            raise FileNotFoundError(
                f"{json_path} が存在しません。--generate を先に実行してください。")
        with open(json_path, encoding="utf-8") as fp:
            jdict = json.load(fp)
        prepare_csv_for_human(jdict, csv_path)


if __name__ == "__main__":
    main()

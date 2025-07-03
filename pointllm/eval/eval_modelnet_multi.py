# pointllm/eval/eval_modelnet_multi.py
import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import re
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import ModelNet
from tqdm import tqdm
from pointllm.eval.evaluator import start_evaluation
from transformers import AutoTokenizer

import os
import json

# ★★★ 複数点群評価用のプロンプトリスト（識別トークンは<pc_1>形式） ★★★
MULTI_PROMPT_LISTS = [
    "<pc_1>\nWhat is this?",                                # index 0 (1 object)
    "<pc_1>\nThis is an object of ",                        # index 1 (1 object)
    "<pc_1> <pc_2>\nWhat is the first object?",            # index 2 (2 objects)
    "<pc_1> <pc_2>\nWhat is the second object?",           # index 3 (2 objects)
    "<pc_1> <pc_2> <pc_3>\nWhat is the first object?",     # index 4 (3 objects)
    "<pc_1> <pc_2> <pc_3>\nWhat is the second object?",    # index 5 (3 objects)
    "<pc_1> <pc_2> <pc_3>\nWhat is the third object?",     # index 6 (3 objects)
    "<pc_1> <pc_2> <pc_3>\nDescribe the third object.",    # index 7 (3 objects)
    "<pc_1> <pc_2>\nCompare the first and second objects.", # index 8 (2 objects)
    "<pc_1> <pc_2> <pc_3>\nWhat are the differences between the second and third objects?",  # index 9 (3 objects)
    "<pc_1> <pc_2> <pc_3>\nOut of these three objects, which object class is typiaclly the largest?",      # index 10 (3 objects)
    "<pc_1> <pc_2> <pc_3>\nWhich object is most suitable for sitting?",  # index 11 (3 objects)
    "<pc_1> <pc_2> <pc_3>\nHow many objects are there? answer in number", # index 12 (3 objects)
    "<pc_1> <pc_2> <pc_3>\nThree objects are inputted, please describe each objects",  # index 13 (3 objects)
    ""
]

# ★★★ プロンプトインデックスと期待される正解オブジェクトの位置の対応 ★★★
PROMPT_TARGET_POSITION = {
    0: 1,   # "What is this?" -> position 1 (最初のオブジェクト)
    1: 1,   # "This is an object of " -> position 1 (最初のオブジェクト)
    2: 1,   # "What is the first object?" -> position 1
    3: 2,   # "What is the second object?" -> position 2
    4: 1,   # "What is the first object?" -> position 13
    5: 2,   # "What is the second object?" -> position 2
    6: 3,   # "What is the third object?" -> position 3
    7: 3,   # "Describe the third object." -> position 3
    8: None,  # "Compare the first and second objects." -> 比較タスク（正解なし）
    9: None,  # "What are the differences between..." -> 比較タスク（正解なし）
    10: None, # "Which object is largest?" -> 判断タスク（正解なし）
    11: None, # "Which object is most suitable for sitting?" -> 判断タスク（正解なし）
    12: None, # "How many objects are there?" -> カウントタスク（正解なし）
    13: None, # "Three objects are inputted, please describe each objects" -> 全体タスク（正解なし）
}

def init_model(args):
    """モデルとトークナイザを初期化する（eval_modelnet_cls.pyと同じ）"""
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:0")  # 使用するGPUを明示的に指定
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        use_cache=True, 
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_multiple_datasets(config_path, split, subset_nums, use_color, data_path=None, num_objects=3):
    """元のデータセットの各サンプルを複数オブジェクトに拡張（順序保持版）"""
    
    if data_path:
        print(f"Loading {split} split of ModelNet datasets from {data_path}.")
        dataset = ModelNet(config_path=config_path, split=split, subset_nums=-1, 
                          use_color=use_color, data_path=data_path)
    else:
        print(f"Loading {split} split of ModelNet datasets.")
        dataset = ModelNet(config_path=config_path, split=split, subset_nums=-1, 
                          use_color=use_color)
    
    print("Creating multi-object combinations (preserving original order)...")
    
    # カテゴリ別にサンプルを整理
    category_samples = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        category = sample['label_names']
        if category not in category_samples:
            category_samples[category] = []
        category_samples[category].append(idx)
    
    categories = list(category_samples.keys())
    
    # 元のデータセットの各サンプルを基準に複数オブジェクトを生成
    multi_samples = []
    
    # subset_nums が指定されている場合はその数まで、そうでなければ全データセット
    if subset_nums > 0:
        target_samples = min(subset_nums, len(dataset))
    else:
        target_samples = len(dataset)
    
    print(f"Generating {target_samples} multi-object combinations...")
    
    for combo_idx in range(target_samples):
        # 元のデータセットから順番にサンプルを取得
        base_sample = dataset[combo_idx]
        base_category = base_sample['label_names']
        
        # 最初のオブジェクトは元のサンプル
        selected_indices = [combo_idx]
        point_clouds_list = [base_sample['point_clouds']]
        labels_list = [base_sample['labels']]
        label_names_list = [base_sample['label_names']]
        
        # 残りのオブジェクトを追加（ランダムだが再現可能）
        random.seed(42 + combo_idx)  # 再現可能なランダム
        for obj_idx in range(1, num_objects):
            # 他のカテゴリからランダムに選択
            available_categories = [cat for cat in categories if cat != base_category]
            if available_categories:
                selected_category = random.choice(available_categories)
            else:
                selected_category = random.choice(categories)
            
            # 選択されたカテゴリからランダムにサンプルを選択
            sample_idx = random.choice(category_samples[selected_category])
            sample = dataset[sample_idx]
            
            selected_indices.append(sample_idx)
            point_clouds_list.append(sample['point_clouds'])
            labels_list.append(sample['labels'])
            label_names_list.append(sample['label_names'])
        
        # 複数点群をテンソルに結合 (M, N, C)
        multi_point_clouds = torch.stack(point_clouds_list, dim=0)  # (M, N, C)
        
        multi_samples.append({
            'indice': combo_idx,
            'original_indices': selected_indices,
            'point_clouds': multi_point_clouds,  # (M, N, C)
            'labels': labels_list,
            'label_names': label_names_list,
            'num_objects': num_objects
        })
    
    print(f"Created {len(multi_samples)} multi-object combinations with {num_objects} objects each.")
    print(f"First object distribution matches original dataset distribution.")
    return multi_samples

def get_multi_dataloader(multi_samples, batch_size, shuffle=False, num_workers=4):
    """マルチオブジェクト用のカスタムDataLoader"""
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    
    class MultiObjectDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    def collate_fn(batch):
        """複数オブジェクト数に対応したcollate関数"""
        # 各サンプルのオブジェクト数を確認
        object_counts = [sample['point_clouds'].shape[0] for sample in batch]
        max_objects = max(object_counts)
        
        # point_cloudsをパディングしてスタック
        padded_point_clouds = []
        for sample in batch:
            pc = sample['point_clouds']
            if pc.shape[0] < max_objects:
                pad_size = max_objects - pc.shape[0]
                padding = torch.zeros(pad_size, pc.shape[1], pc.shape[2], dtype=pc.dtype)
                padded_pc = torch.cat([pc, padding], dim=0)
                padded_point_clouds.append(padded_pc)
            else:
                padded_point_clouds.append(pc)
        
        # その他のフィールドをリストとして収集
        batched_data = {
            'point_clouds': torch.stack(padded_point_clouds),
            'labels': [sample['labels'] for sample in batch],
            'label_names': [sample['label_names'] for sample in batch],
            'indice': [sample['indice'] for sample in batch],
            'num_objects': [sample['num_objects'] for sample in batch]
        }
        return batched_data
    
    dataset = MultiObjectDataset(multi_samples)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    """推論関数（eval_modelnet_cls.pyと同じハイパーパラメータ）"""
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    # バッチ処理後にキャッシュを明示的にクリア
    torch.cuda.empty_cache()

    return outputs

def start_multi_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file):
    """マルチオブジェクトの推論を実行（最終修正版）"""
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    qs = MULTI_PROMPT_LISTS[prompt_index]
    num_objects = qs.count('<pc_')
    target_position = PROMPT_TARGET_POSITION.get(prompt_index, None)
    print(f"[INFO] Using prompt: '{qs}' for {num_objects} objects")
    print(f"[INFO] Target position: {target_position}")
    
    results = {"prompt": qs, "num_objects": num_objects, "target_position": target_position}
    
    # ★★★ ここに前処理を追加 ★★★
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    replace_token = point_backbone_config['default_point_patch_token'] * point_token_len
    if point_backbone_config['mm_use_point_start_end']:
        replace_token = point_backbone_config['default_point_start_token'] + replace_token + point_backbone_config['default_point_end_token']

    # プロンプト内の全ての識別子を、モデル用の内部表現に置換する
    processed_qs = qs
    # <pc_数字> パターンを全て見つける
    pc_placeholders = re.findall(r'<pc_\d+>', processed_qs)
    print(f"[INFO] Found {len(pc_placeholders)} point cloud placeholders: {pc_placeholders}")
    # 見つけたプレースホルダーを順番に置換していく
    for placeholder in pc_placeholders:
        # replaceの第三引数に1を指定し、最初に見つかったものだけを置換（順序を保つため）
        processed_qs = processed_qs.replace(placeholder, replace_token, 1)
    
    print(f"[INFO] Original prompt: {qs}")
    print(f"[INFO] Processed prompt: {processed_qs}")
    # ★★★ ここまで修正 ★★★

    # 会話テンプレートの準備
    conv.messages = []  # 会話履歴をリセット
    # ★ 処理済みのプロンプトを使用
    conv.append_message(conv.roles[0], processed_qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt], return_tensors="pt")
    input_ids_template = inputs.input_ids.cuda()

    stopping_criteria = KeywordsStoppingCriteria([stop_str, tokenizer.eos_token], tokenizer, input_ids_template)

    responses = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # PyTorchのデフォルトcollate_fnは辞書を返す
        point_clouds = batch['point_clouds'].cuda().to(model.dtype)
        batch_size = point_clouds.shape[0]
        
        input_ids = input_ids_template.repeat(batch_size, 1)

        # バッチ化されたテンソルを直接モデルに渡す
        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria)

        # バッチ化された結果を正しく処理するループ
        for i in range(batch_size):
            # batch['label_names']はリストのリスト [[name1, name2, ...], [name1, name2, ...], ...]
            # batch['labels']はテンソルのリスト [tensor([label1, label2, ...]), tensor([label1, label2, ...]), ...]
            sample_label_names = batch["label_names"][i]
            sample_labels = batch["labels"][i]
            
            object_details = [
                {"position": j + 1, "category": sample_label_names[j], "label": sample_labels[j].item() if torch.is_tensor(sample_labels[j]) else sample_labels[j]}
                for j in range(len(sample_label_names))
            ]

            object_id = batch["indice"][i].item() if torch.is_tensor(batch["indice"][i]) else batch["indice"][i]

            responses.append({
                "object_id": object_id,
                "object_details": object_details,
                "model_output": outputs[i],
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    # 乱数シードを固定（再現性のため）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model_basename = os.path.basename(os.path.expanduser(args.model_name))

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_name, "evaluation_multi")
        
    if args.data_path:
        variant_name = os.path.basename(os.path.dirname(args.data_path))
        args.output_file = f"ModelNet_multi_classification_{variant_name}_prompt{args.prompt_index}_obj{args.num_objects}_batch{args.batch_size}.json"
    else:
        args.output_file = f"ModelNet_multi_classification_prompt{args.prompt_index}_obj{args.num_objects}_batch{args.batch_size}.json"
    
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # 評価設定の表示
    # * First inferencing, then evaluate

    if not os.path.exists(args.output_file_path):
        # * need to generate results first
        multi_samples = load_multiple_datasets(
            config_path=args.config_path, 
            split=args.split, 
            subset_nums=args.subset_nums, 
            use_color=args.use_color, 
            data_path=args.data_path,
            num_objects=args.num_objects
        )
        dataloader = get_multi_dataloader(multi_samples, args.batch_size, args.shuffle, args.num_workers)
    
        model, tokenizer, conv = init_model(args)

        # * ouptut
        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_multi_generation(model, tokenizer, conv, dataloader, args.prompt_index, args.output_dir, args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    # * evaluation file
    evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    # * start evaluation
    if args.start_eval:
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type="modelnet-close-set-classification", model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None,
        help="YAML for ModelNet")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Output directory for results")
    parser.add_argument("--data_path", type=str, default="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat",
        help="Path to ModelNet data file")
    parser.add_argument("--model_name", type=str,
        default="./outputs/PointLLM_train_stage2/PointLLM_train_stage2_naive_batch")
    
    # * dataset type
    parser.add_argument("--split", type=str, default="test", help="train or test.")
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--subset_nums", type=int, default=-1) # * only use "subset_nums" of samples, mainly for debug 
    parser.add_argument("--num_objects", type=int, default=1, help="Number of objects per sample")

    # Evaluation settings
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-3.5-turbo-0613", 
                       choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], 
                       help="Type of the model used to evaluate.")

    args = parser.parse_args()

    main(args) 
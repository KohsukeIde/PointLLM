# pointllm/eval/eval_modelnet_multi.py
import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
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

# PEFT imports for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT library not available. LoRA evaluation disabled.")
    PEFT_AVAILABLE = False

# 複数点群用のプロンプトリスト（識別トークン付き）
MULTI_PROMPT_LISTS = [
    "<pcA> What is this?",                          # index 0 (1 object)
    "<pcA> This is an object of ",                  # index 1 (1 object)
    "<pcA> <pcB> What is the first object?",        # index 2 (2 objects)
    "<pcA> <pcB> What is the second object?",       # index 3 (2 objects)
    "<pcA> <pcB> <pcC> What is the first object?",  # index 4 (3 objects)
    "<pcA> <pcB> <pcC> What is the second object?", # index 5 (3 objects)
    "<pcA> <pcB> <pcC> What is the third object?",  # index 6 (3 objects)
    "<pcA> <pcB> <pcC> Describe the third object.", # index 7 (3 objects)
    "<pcA> <pcB> Compare the first and second objects.",  # index 8 (2 objects)
    "<pcA> <pcB> <pcC> What are the differences between the second and third objects?",  # index 9 (3 objects)
    "<pcA> <pcB> <pcC> Which object is largest?",   # index 10 (3 objects)
    "<pcA> <pcB> <pcC> Which object is most suitable for sitting?"  # index 11 (3 objects)
]

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    print(f'[INFO] Base model name: {os.path.basename(model_name)}')

    # トークナイザー読み込み（LoRAディレクトリ優先）
    tokenizer = None
    if hasattr(args, 'lora_dir') and args.lora_dir and os.path.isdir(args.lora_dir):
        try:
            print(f'[INFO] Loading tokenizer from LoRA directory: {args.lora_dir}')
            tokenizer = AutoTokenizer.from_pretrained(args.lora_dir)
            print('[INFO] Successfully loaded tokenizer from LoRA directory')
        except Exception as e:
            print(f"Failed to load tokenizer from LoRA directory {args.lora_dir}: {e}")
    
    # フォールバック: ベースモデルからトークナイザーを読み込み
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Failed to load tokenizer from {model_name}: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
                print("Using fallback huggyllama tokenizer")
            except Exception as e2:
                tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
                print("Using fallback decapoda tokenizer")
    
    # pad_tokenの設定（必要に応じて）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print('[INFO] Set pad_token to eos_token')

    device = torch.device("cuda:0")
    
    # ベースモデル読み込み（堅牢化）
    try:
        base_model = PointLLMLlamaForCausalLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=True,
            use_cache=True, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        print(f"Successfully loaded model from: {model_name}")
    except Exception as e:
        print(f"Failed to load model from {model_name}: {e}")
        # フォールバック処理
        cache_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--RunsenXu--PointLLM_7B_v1.2/snapshots/37d8c15aaff8e7e05f04729dcb6960d5758e9f86")
        local_model_path = "/groups/gag51402/RunsenXu/PointLLM_7B_v1.2"
        
        for fallback_path in [cache_model_path, local_model_path]:
            if os.path.exists(fallback_path):
                try:
                    print(f"Trying fallback path: {fallback_path}")
                    base_model = PointLLMLlamaForCausalLM.from_pretrained(
                        fallback_path,
                        low_cpu_mem_usage=True,
                        use_cache=True, 
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    ).to(device)
                    print(f"Successfully loaded model from: {fallback_path}")
                    break
                except Exception as e_fallback:
                    print(f"Failed to load from {fallback_path}: {e_fallback}")
                    continue
        else:
            raise ValueError("Could not load PointLLM model from any available path")
    
    # LoRAモデルの読み込み（指定された場合）
    if hasattr(args, 'lora_dir') and args.lora_dir and PEFT_AVAILABLE:
        print(f'[INFO] Loading LoRA adapter from: {args.lora_dir}')
        try:
            model = PeftModel.from_pretrained(base_model, args.lora_dir)
            print('[INFO] LoRA adapter loaded successfully')
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print('[WARNING] Size mismatch detected, loading with ignore_mismatched_sizes=True')
                model = PeftModel.from_pretrained(base_model, args.lora_dir, ignore_mismatched_sizes=True)
                print('[INFO] LoRA adapter loaded with size mismatch ignored')
            else:
                raise e
        
        if hasattr(args, 'merge_lora') and args.merge_lora:
            print('[INFO] Merging LoRA weights into base model...')
            model = model.merge_and_unload()
    else:
        model = base_model
        if hasattr(args, 'lora_dir') and args.lora_dir:
            print('[WARNING] LoRA directory specified but PEFT not available')
    
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    # カスタムトークンが追加されている場合、埋め込み行列を拡張
    if hasattr(args, 'lora_dir') and args.lora_dir:
        custom_tokens_file = os.path.join(args.lora_dir, 'custom_tokens.json')
        if os.path.exists(custom_tokens_file):
            print('[INFO] Loading custom tokens information...')
            with open(custom_tokens_file, 'r') as f:
                custom_tokens_info = json.load(f)
            
            expected_vocab_size = custom_tokens_info.get('tokenizer_size', len(tokenizer))
            current_vocab_size = model.get_input_embeddings().weight.size(0)
            
            print(f'[INFO] Current vocab size: {current_vocab_size}')
            print(f'[INFO] Expected vocab size: {expected_vocab_size}')
            
            if expected_vocab_size > current_vocab_size:
                print(f'[INFO] Resizing model embeddings from {current_vocab_size} to {expected_vocab_size}')
                model.resize_token_embeddings(expected_vocab_size)
                print('[INFO] Model embeddings resized successfully')
            
            # カスタムトークンIDの確認
            cls_token_ids = custom_tokens_info.get('cls_token_ids', {})
            for token, token_id in cls_token_ids.items():
                actual_id = tokenizer.convert_tokens_to_ids(token)
                print(f'[INFO] Token {token}: expected ID {token_id}, actual ID {actual_id}')

    # 推論専用では PointNet を固定する
    model.get_model().fix_pointnet = True

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
    """マルチオブジェクト用のカスタムDataLoader（高速化版）"""
    
    class MultiObjectDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    def collate_fn(batch):
        """バッチ処理用のcollate関数"""
        if len(batch) == 1:
            return batch[0]
        else:
            # 複数サンプルのバッチ処理
            batch_data = {
                'indice': [item['indice'] for item in batch],
                'original_indices': [item['original_indices'] for item in batch],
                'point_clouds': [item['point_clouds'] for item in batch],
                'labels': [item['labels'] for item in batch],
                'label_names': [item['label_names'] for item in batch],
                'num_objects': [item['num_objects'] for item in batch]
            }
            return batch_data
    
    dataset = MultiObjectDataset(multi_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dataloader

def generate_outputs_batch(model, tokenizer, input_ids, point_clouds_batch, stopping_criteria, 
                          do_sample=True, temperature=1.0, max_length=2048):
    """バッチ処理対応の高速推論関数"""
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds_batch,
            do_sample=do_sample,
            temperature=temperature,
            top_k=50,
            max_length=max_length,
            top_p=0.95,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]
    
    # デバッグ情報（最初のサンプルのみ）
    if len(outputs) > 0:
        output_len = len(outputs[0].split())
        input_len = input_ids.shape[1]
        generated_len = output_ids.shape[1] - input_len
        
        print(f"[DEBUG] Input length: {input_len} tokens")
        print(f"[DEBUG] Generated length: {generated_len} tokens")
        print(f"[DEBUG] Output word count: {output_len} words")
        
        if output_len > 50:  # 長すぎる出力の警告
            print(f"[WARNING] Long output detected!")
            print(f"[DEBUG] First 200 chars: {outputs[0][:200]}...")
            print(f"[DEBUG] Last 200 chars: ...{outputs[0][-200:]}")
            
            # 繰り返しパターンの検出
            words = outputs[0].split()
            if len(words) > 10:
                # 最後の10単語が繰り返されているかチェック
                last_10 = ' '.join(words[-10:])
                prev_10 = ' '.join(words[-20:-10]) if len(words) >= 20 else ""
                if last_10 == prev_10:
                    print(f"[WARNING] Repetition detected: '{last_10}'")

    return outputs

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    # メモリクリア
    torch.cuda.empty_cache()

    return outputs

def generate_outputs_with_features(model, tokenizer, input_ids, point_features_list, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    """特徴量リストを使って直接LLM推論を実行"""
    model.eval()
    
    # 入力埋め込みを取得
    inputs_embeds = model.get_input_embeddings()(input_ids)
    
    # 点群特徴量をリスト形式で渡す（pointllm.pyの処理をバイパス）
    point_backbone_config = model.get_model().point_backbone_config
    
    # 累積的置換ロジック（pointllm.pyから）
    new_input_embeds = []
    for batch_idx, cur_input_embeds in enumerate(inputs_embeds):
        cur_input_ids = input_ids[batch_idx]
        
        if point_backbone_config['mm_use_point_start_end']:
            # <point_start>...<point_end> 形式の処理
            point_start_positions = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
            point_end_positions = torch.where(cur_input_ids == point_backbone_config["point_end_token"])[0]
            
            if len(point_start_positions) != len(point_end_positions):
                raise ValueError("The number of point start tokens and point end tokens should be the same.")
            
            if len(point_start_positions) > len(point_features_list):
                raise ValueError(f"Found {len(point_start_positions)} point regions but only {len(point_features_list)} point clouds.")
            
            # 累積的置換: 後ろから前に向かって処理（順序修正）
            remaining_features = point_features_list.copy()
            for i in reversed(range(len(point_start_positions))):
                start_pos = point_start_positions[i]
                end_pos = point_end_positions[i]
                # 修正: pop()ではなくpop(0)を使用して正しい順序を保持
                point_feature = remaining_features.pop(0).to(device=cur_input_embeds.device)
                
                # 置換実行: <point_start> + point_features + <point_end>
                replacement = torch.cat([
                    cur_input_embeds[start_pos:start_pos+1],  # <point_start>
                    point_feature,
                    cur_input_embeds[end_pos:end_pos+1]       # <point_end>
                ], dim=0)
                
                # 元の埋め込みを更新
                before_part = cur_input_embeds[:start_pos]
                after_part = cur_input_embeds[end_pos+1:]
                
                cur_input_embeds = torch.cat([
                    before_part,
                    replacement,
                    after_part
                ], dim=0)
        
        new_input_embeds.append(cur_input_embeds)
    
    # シーケンス長チェック
    if len(new_input_embeds) > 1:
        seq_lengths = [emb.shape[0] for emb in new_input_embeds]
        if not all(length == seq_lengths[0] for length in seq_lengths):
            raise RuntimeError(f"Sequence length mismatch: {seq_lengths}")
    
    inputs_embeds = torch.stack(new_input_embeds, dim=0)
    
    # attention_maskを生成（全て1で埋める）
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    
    # LLM推論実行（点群処理は既に完了しているため、point_cloudsは渡さない）
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=None,  # inputs_embedsを使うのでNone
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,  # attention_maskを追加
            point_clouds=None,  # 既に処理済みなのでNone
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria])

    input_token_len = inputs_embeds.shape[1]
    n_diff_input_output = (inputs_embeds.shape[1] != output_ids.shape[1] - input_token_len)
    if n_diff_input_output:
        print(f'[Warning] Output length mismatch')
    
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    # メモリクリア
    torch.cuda.empty_cache()

    return outputs

def start_multi_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file, num_objects=3, batch_size=1):
    """高速化されたマルチオブジェクト推論関数"""
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    # プロンプトと点群数の整合性チェック
    qs = MULTI_PROMPT_LISTS[prompt_index]
    expected_objects = qs.count('<pc')  # プロンプト内の識別トークン数をカウント
    
    if expected_objects != num_objects:
        print(f"[WARNING] Prompt-object mismatch detected!")
        print(f"  Prompt index {prompt_index}: '{qs}'")
        print(f"  Expected objects: {expected_objects}")
        print(f"  Actual objects: {num_objects}")
        
        # 自動修正: 点群数に応じて適切なプロンプトを選択
        if num_objects == 1:
            prompt_index = 0  # "<pcA> What is this?"
        elif num_objects == 2:
            if "second object" in qs:
                prompt_index = 3  # "<pcA> <pcB> What is the second object?"
            else:
                prompt_index = 2  # "<pcA> <pcB> What is the first object?"
        elif num_objects == 3:
            if "first object" in qs:
                prompt_index = 4  # "<pcA> <pcB> <pcC> What is the first object?"
            elif "second object" in qs:
                prompt_index = 5  # "<pcA> <pcB> <pcC> What is the second object?"
            elif "third object" in qs:
                prompt_index = 6  # "<pcA> <pcB> <pcC> What is the third object?"
            else:
                prompt_index = 7  # "<pcA> <pcB> <pcC> Describe the third object."
        
        qs = MULTI_PROMPT_LISTS[prompt_index]
        print(f"[INFO] Auto-corrected to prompt index {prompt_index}: '{qs}'")
    
    print(f"[INFO] Using prompt: '{qs}' for {num_objects} objects")

    results = {"prompt": qs, "num_objects": num_objects}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    # 複数点群のプロンプト生成（学習時と同じ順序：識別トークン → 点群ブロック）
    if mm_use_point_start_end:
        # 識別トークンのリスト
        position_tokens = ["<pcA>", "<pcB>", "<pcC>"]
        
        # 各オブジェクトに対して：識別トークン + 点群ブロック
        tokens_with_id = ""
        for i in range(num_objects):
            if i < len(position_tokens):
                id_token = position_tokens[i]
            else:
                # 3個以上の場合は拡張（必要に応じて）
                id_token = f"<pc{chr(ord('A') + i)}>"
            
            tokens_with_id += (
                id_token + " " +                             # ① 識別トークン
                default_point_start_token +                  # ② <point_start>
                default_point_patch_token * point_token_len + # ③ 513 patch
                default_point_end_token + " "                # ④ <point_end>
            )
        
        # 元のプロンプトから識別トークンを除去（重複を避けるため）
        clean_prompt = qs
        for token in position_tokens:
            clean_prompt = clean_prompt.replace(token + " ", "")
        
        # 最終的なプロンプト：識別トークン付き点群ブロック + 質問
        qs = tokens_with_id + clean_prompt
    else:
        # <point_patch> モードは複数点群非対応
        raise ValueError("Multiple point clouds are not supported with point_patch mode. Use mm_use_point_start_end=True.")
    
    # プロンプトを省略表記に変換（既存JSONフォーマットとの一致のため）
    simplified_prompt = qs
    if mm_use_point_start_end:
        # <point_patch>の繰り返し部分を省略表記に変換
        patch_pattern = default_point_patch_token * point_token_len
        simplified_pattern = f"{default_point_patch_token}*{point_token_len}"
        simplified_prompt = simplified_prompt.replace(patch_pattern, simplified_pattern)
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids_ = torch.as_tensor(inputs.input_ids).cuda()
    
    # 強化されたStopping Criteria
    stop_words = [stop_str]
    if stop_str != tokenizer.eos_token:
        stop_words.append(tokenizer.eos_token)
    # 繰り返しパターンも停止条件に追加
    stop_words.extend(["3D rendering", "3D model", ". 3D", "Black and white"])
    
    stopping_criteria = KeywordsStoppingCriteria(stop_words, tokenizer, input_ids_)

    responses = []
    
    # プログレスバー追加
    progress_bar = tqdm(dataloader, desc="Evaluating", total=len(dataloader))

    for batch in progress_bar:
        try:
            # バッチサイズ1の場合（従来の処理）
            if isinstance(batch["indice"], (int, torch.Tensor)) and not isinstance(batch["indice"], list):
                # 単一サンプル処理
                multi_point_clouds = batch["point_clouds"].squeeze(0).cuda().to(model.dtype)
                labels = batch["labels"]
                label_names = batch["label_names"]
                indice = batch["indice"]
                original_indices = batch["original_indices"]

                # 4Dテンソル形式に変換
                multi_point_clouds_4d = multi_point_clouds.unsqueeze(0)  # (1, M, N, C)
                input_ids = input_ids_.clone()
                
                outputs = generate_outputs_batch(model, tokenizer, input_ids, multi_point_clouds_4d, stopping_criteria)

                # 結果処理（単一サンプル）
                object_details = []
                for obj_idx in range(num_objects):
                    object_details.append({
                        "position": obj_idx + 1,
                        "category": label_names[obj_idx] if isinstance(label_names[obj_idx], list) else [label_names[obj_idx]],  # リスト形式に統一
                        "label": labels[obj_idx].item() if torch.is_tensor(labels[obj_idx]) else labels[obj_idx],
                        "original_dataset_index": original_indices[obj_idx].item() if torch.is_tensor(original_indices[obj_idx]) else original_indices[obj_idx]
                    })

                responses.append({
                    "object_id": indice.item() if torch.is_tensor(indice) else indice,
                    "object_details": object_details,
                    "model_output": outputs[0],
                    "prompt": simplified_prompt,  # 省略表記を使用
                    "num_objects": num_objects
                })
                
            else:
                # バッチサイズ>1の場合（新しい高速処理）
                batch_size_actual = len(batch["indice"])
                
                # 点群データのパディング処理
                batch_point_clouds = []
                max_objects = max(pc.shape[0] for pc in batch["point_clouds"])
                
                for pc in batch["point_clouds"]:
                    current_objects = pc.shape[0]
                    if current_objects < max_objects:
                        # ゼロパディング
                        pad_objects = max_objects - current_objects
                        padding = torch.zeros(pad_objects, pc.shape[1], pc.shape[2], dtype=pc.dtype)
                        padded_pc = torch.cat([pc, padding], dim=0)
                        batch_point_clouds.append(padded_pc)
                    else:
                        batch_point_clouds.append(pc)
                
                # バッチテンソルに変換
                batch_point_clouds_tensor = torch.stack(batch_point_clouds).cuda().to(model.dtype)  # (B, M, N, C)
                
                # 入力IDを複製
                input_ids_batch = input_ids_.repeat(batch_size_actual, 1)
                
                # バッチ推論
                outputs = generate_outputs_batch(model, tokenizer, input_ids_batch, batch_point_clouds_tensor, stopping_criteria)
                
                # 結果処理（バッチ）
                for i in range(batch_size_actual):
                    object_details = []
                    for obj_idx in range(num_objects):
                        if obj_idx < len(batch["label_names"][i]):  # パディング部分を除外
                            object_details.append({
                                "position": obj_idx + 1,
                                "category": batch["label_names"][i][obj_idx] if isinstance(batch["label_names"][i][obj_idx], list) else [batch["label_names"][i][obj_idx]],  # リスト形式に統一
                                "label": batch["labels"][i][obj_idx].item() if torch.is_tensor(batch["labels"][i][obj_idx]) else batch["labels"][i][obj_idx],
                                "original_dataset_index": batch["original_indices"][i][obj_idx].item() if torch.is_tensor(batch["original_indices"][i][obj_idx]) else batch["original_indices"][i][obj_idx]
                            })

                    responses.append({
                        "object_id": batch["indice"][i].item() if torch.is_tensor(batch["indice"][i]) else batch["indice"][i],
                        "object_details": object_details,
                        "model_output": outputs[i],
                        "prompt": simplified_prompt,  # 省略表記を使用
                        "num_objects": num_objects
                    })
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'processed': len(responses),
                'batch_size': batch_size_actual if 'batch_size_actual' in locals() else 1
            })
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

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
    print("=" * 50)
    print("LoRA位置参照評価設定")
    print("=" * 50)
    print(f"バッチサイズ: {args.batch_size}")
    print(f"ワーカー数: {args.num_workers}")
    print(f"生成パラメータ: do_sample=True, temperature=1.0, top_k=50, top_p=0.95")
    print(f"最大長: 2048 tokens (通常のModelNet評価と同じ)")
    print(f"繰り返しペナルティ: 1.1")
    print(f"LoRAディレクトリ: {args.lora_dir}")
    print("=" * 50)

    # 結果生成または読み込み
    if not os.path.exists(args.output_file_path):
        # マルチオブジェクトデータセット作成
        multi_samples = load_multiple_datasets(
            config_path=args.config_path, 
            split=args.split, 
            subset_nums=args.subset_nums, 
            use_color=args.use_color, 
            data_path=args.data_path,
            num_objects=args.num_objects
        )
        
        # 高速化DataLoader
        dataloader = get_multi_dataloader(
            multi_samples, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )
    
        model, tokenizer, conv = init_model(args)

        print(f'[INFO] Start generating results for {args.output_file}.')
        print(f'[INFO] 予想処理時間: {len(multi_samples) // (args.batch_size * 60):.1f}分 (バッチサイズ{args.batch_size}で毎分60バッチ想定)')
        
        results = start_multi_generation(
            model, tokenizer, conv, dataloader, args.prompt_index, 
            args.output_dir, args.output_file, args.num_objects, args.batch_size
        )

        # メモリ解放
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    print(f"[INFO] Multi-object evaluation completed with {len(results.get('results', []))} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None,
        help="YAML for ModelNet")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Output directory for results")
    parser.add_argument("--data_path", type=str, default="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat",
        help="Path to ModelNet data file")
    parser.add_argument("--model_name", type=str, 
        default="RunsenXu/PointLLM_7B_v1.2")
    
    # LoRA settings
    parser.add_argument("--lora_dir", type=str, default=None,
        help="Path to LoRA adapter directory")
    parser.add_argument("--merge_lora", action="store_true", default=False,
        help="Merge LoRA weights into base model")

    # Dataset settings
    parser.add_argument("--split", type=str, default="test", help="train or test.")
    parser.add_argument("--use_color", action="store_true", default=True)
    parser.add_argument("--subset_nums", type=int, default=-1, help="Number of samples to evaluate. -1 for dataset size")
    parser.add_argument("--num_objects", type=int, default=2, help="Number of objects per sample")

    # 高速化設定
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation (高速化)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers (高速化)")

    # Evaluation settings
    parser.add_argument("--prompt_index", type=int, default=3)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-3.5-turbo-0613", 
                       choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], 
                       help="Type of the model used to evaluate.")

    args = parser.parse_args()

    main(args)
# pointllm/eval/simple_lora_test.py
import argparse
import torch
import os
import random
import numpy as np
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import ModelNet
from transformers import AutoTokenizer

# PEFT imports for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT library not available. LoRA evaluation disabled.")
    PEFT_AVAILABLE = False

def init_model(lora_dir=None):
    """ベースモデルまたはLoRA適用モデルを初期化"""
    disable_torch_init()
    
    # トークナイザー読み込み
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        except Exception as e2:
            raise ValueError("Could not load any compatible Llama tokenizer")
    
    device = torch.device("cuda:0")
    
    # ベースモデル読み込み
    cache_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--RunsenXu--PointLLM_7B_v1.2/snapshots/37d8c15aaff8e7e05f04729dcb6960d5758e9f86")
    
    print("Loading PointLLM base model...")
    base_model = PointLLMLlamaForCausalLM.from_pretrained(
        cache_model_path,
        low_cpu_mem_usage=True,
        use_cache=True, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    # LoRA適用（指定された場合）
    if lora_dir and PEFT_AVAILABLE:
        print(f'Loading LoRA adapter from: {lora_dir}')
        model = PeftModel.from_pretrained(base_model, lora_dir)
        model_type = "LoRA"
    else:
        model = base_model
        model_type = "Base"
    
    print(f'Model type: {model_type}')
    
    # PointLLM初期化
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.get_model().fix_pointnet = True

    # 会話テンプレート
    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv, model_type

def create_simple_test():
    """簡単なテストケースを作成"""
    
    print("Loading ModelNet test dataset...")
    try:
        dataset = ModelNet(config_path=None, split="test", subset_nums=100,  # 最初の100サンプルのみ読み込み
                          use_color=True, data_path="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat")
        print(f"Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # 異なるカテゴリから2つのサンプルを選択
    categories = ["chair", "table", "sofa", "bed", "desk"]
    selected_samples = []
    
    print("Searching for samples from specific categories...")
    for i in range(min(len(dataset), 50)):  # 最初の50サンプルから探索
        sample = dataset[i]
        print(f"Sample {i}: {sample['label_names']}")
        if sample['label_names'] in categories and len(selected_samples) < 2:
            selected_samples.append(sample)
            print(f"Selected sample {i}: {sample['label_names']}")
        if len(selected_samples) >= 2:
            break
    
    if len(selected_samples) < 2:
        # フォールバック：最初の2つのサンプルを使用
        print("Using first 2 samples as fallback")
        selected_samples = [dataset[0], dataset[1]]
    
    # 複数点群をテンソルに結合
    point_clouds_list = [s['point_clouds'] for s in selected_samples]
    multi_point_clouds = torch.stack(point_clouds_list, dim=0)  # (M, N, C)
    
    test_case = {
        'point_clouds': multi_point_clouds,
        'label_names': [s['label_names'] for s in selected_samples],
        'num_objects': len(selected_samples)
    }
    
    print(f"Created test case with objects: {test_case['label_names']}")
    return test_case

def test_model(model, tokenizer, conv, test_case, model_type):
    """単一のテストケースでモデルを評価"""
    
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    
    # テストプロンプト
    test_prompts = [
        "What is the first object?",
        "What is the second object?",
        "List all objects from first to last."
    ]
    
    results = []
    
    for prompt in test_prompts:
        print(f"\nTesting {model_type} model with prompt: '{prompt}'")
        
        try:
            # プロンプトの準備
            conv_copy = conv.copy()
            num_objects = test_case['num_objects']
            
            # 複数点群用のプロンプト生成
            point_tokens = ""
            for i in range(num_objects):
                point_tokens += default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + " "
            
            full_prompt = point_tokens + prompt
            
            conv_copy.append_message(conv_copy.roles[0], full_prompt)
            conv_copy.append_message(conv_copy.roles[1], None)
            
            formatted_prompt = conv_copy.get_prompt()
            inputs = tokenizer([formatted_prompt])
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
            
            # 停止基準
            stop_str = conv_copy.sep if conv_copy.sep_style != SeparatorStyle.TWO else conv_copy.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            
            # 推論実行
            point_clouds = test_case['point_clouds'].unsqueeze(0).cuda().to(model.dtype)  # (1, M, N, C)
            
            model.eval()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=True,
                    temperature=0.2,
                    top_k=50,
                    max_new_tokens=50,
                    top_p=0.9,
                    stopping_criteria=[stopping_criteria]
                )
            
            input_token_len = input_ids.shape[1]
            output = tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=True).strip()
            
            print(f"Output: {output}")
            print(f"Ground truth: {test_case['label_names']}")
            
            # 結果保存
            result = {
                "prompt": prompt,
                "model_output": output,
                "ground_truth": test_case['label_names'],
                "model_type": model_type
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            continue
    
    return results

def compare_results(base_results, lora_results):
    """結果を比較"""
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for i, prompt in enumerate(["What is the first object?", "What is the second object?", "List all objects from first to last."]):
        if i < len(base_results) and i < len(lora_results):
            print(f"\nPrompt: '{prompt}'")
            print(f"Ground Truth: {base_results[i]['ground_truth']}")
            print(f"Base Model Output: {base_results[i]['model_output']}")
            print(f"LoRA Model Output: {lora_results[i]['model_output']}")
            
            # 簡単な評価
            base_output = base_results[i]['model_output'].lower()
            lora_output = lora_results[i]['model_output'].lower()
            ground_truth = [gt.lower() for gt in base_results[i]['ground_truth']]
            
            base_correct = False
            lora_correct = False
            
            if "first" in prompt.lower() and len(ground_truth) > 0:
                base_correct = ground_truth[0] in base_output
                lora_correct = ground_truth[0] in lora_output
            elif "second" in prompt.lower() and len(ground_truth) > 1:
                base_correct = ground_truth[1] in base_output
                lora_correct = ground_truth[1] in lora_output
            else:
                base_correct = any(gt in base_output for gt in ground_truth)
                lora_correct = any(gt in lora_output for gt in ground_truth)
            
            print(f"Base Model Correct: {base_correct}")
            print(f"LoRA Model Correct: {lora_correct}")
            if lora_correct and not base_correct:
                print("✅ LoRA improved the result!")
            elif base_correct and not lora_correct:
                print("❌ LoRA made the result worse")
            elif base_correct and lora_correct:
                print("✅ Both models correct")
            else:
                print("❌ Both models incorrect")

def main():
    parser = argparse.ArgumentParser(description="Simple test of Base vs LoRA PointLLM")
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # シード固定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("Simple LoRA vs Base Model Test")
    print("="*40)
    
    # テストケース作成
    test_case = create_simple_test()
    
    # ベースモデル評価
    print("\n--- Base Model Test ---")
    base_model, base_tokenizer, base_conv, _ = init_model(lora_dir=None)
    base_results = test_model(base_model, base_tokenizer, base_conv, test_case, "Base")
    
    # メモリクリア
    del base_model, base_tokenizer
    torch.cuda.empty_cache()
    
    # LoRAモデル評価
    print("\n--- LoRA Model Test ---")
    lora_model, lora_tokenizer, lora_conv, _ = init_model(lora_dir=args.lora_dir)
    lora_results = test_model(lora_model, lora_tokenizer, lora_conv, test_case, "LoRA")
    
    # メモリクリア
    del lora_model, lora_tokenizer
    torch.cuda.empty_cache()
    
    # 結果比較
    compare_results(base_results, lora_results)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 
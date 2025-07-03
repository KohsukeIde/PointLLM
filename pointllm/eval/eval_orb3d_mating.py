# pointllm/eval/eval_orb3d_mating.py
import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import pandas as pd
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data.utils import farthest_point_sample, pc_normalize
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import glob

# Shape Mating用のプロンプトリスト
BINARY_MATING_PROMPTS = [
    "Does the first object fit with the second object?",
    "Can these two parts be assembled together?",
    "Do these objects mate perfectly?",
    "Are these parts complementary?",
    "Can the first part connect to the second part?"
]

MULTI_CHOICE_PROMPTS = [
    "Which object (0, 1, or 2) mates perfectly with the first object?",
    "Which part (0, 1, or 2) can be assembled with the first object?",
    "Which object (0, 1, or 2) is the perfect match for the first object?",
    "Select the matching part for the first object: 0, 1, or 2?",
    "Which object (0, 1, or 2) complements the first object?"
]

class ORB3DShapeMatingDataset:
    def __init__(self, data_path, npoints=8192, split_ratio=0.8, test_type="binary"):
        """
        Args:
            data_path: ORB3Dデータセットのルートパス
            npoints: サンプリング点数
            split_ratio: 訓練/テスト分割比率
            test_type: "binary" or "multi_choice"
        """
        self.data_path = data_path
        self.npoints = npoints
        self.test_type = test_type
        
        # オブジェクトディレクトリを収集
        self.object_dirs = [d for d in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, d))]
        
        # データセット分割
        random.shuffle(self.object_dirs)
        split_idx = int(len(self.object_dirs) * split_ratio)
        self.train_objects = self.object_dirs[:split_idx]
        self.test_objects = self.object_dirs[split_idx:]
        
        print(f"Total objects: {len(self.object_dirs)}")
        print(f"Train objects: {len(self.train_objects)}")
        print(f"Test objects: {len(self.test_objects)}")
        
        # テストサンプルを生成
        self.test_samples = self._generate_test_samples()
        
    def _load_point_cloud(self, csv_path):
        """CSVファイルから点群を読み込み、前処理を適用"""
        df = pd.read_csv(csv_path, header=None)
        points = df.values.astype(np.float32)
        
        # FPSでサンプリング
        if points.shape[0] > self.npoints:
            points = farthest_point_sample(points, self.npoints)
        elif points.shape[0] < self.npoints:
            # 点数が不足している場合は繰り返し
            indices = np.random.choice(points.shape[0], self.npoints, replace=True)
            points = points[indices]
        
        # 正規化
        points = pc_normalize(points)
        
        return torch.from_numpy(points).float()
    
    def _get_mating_pairs(self, object_id):
        """指定オブジェクトの全てのmating pairを取得"""
        object_path = os.path.join(self.data_path, object_id)
        pairs = []
        
        # shell/planar と shell/parabolic を探索
        for cut_type in ["planar", "parabolic"]:
            cut_dir = os.path.join(object_path, "shell", cut_type)
            if not os.path.exists(cut_dir):
                continue
                
            # 各バリエーション（0-9）を探索
            for var_dir in os.listdir(cut_dir):
                var_path = os.path.join(cut_dir, var_dir)
                if not os.path.isdir(var_path):
                    continue
                    
                partA_path = os.path.join(var_path, "partA-pc.csv")
                partB_path = os.path.join(var_path, "partB-pc.csv")
                
                if os.path.exists(partA_path) and os.path.exists(partB_path):
                    pairs.append({
                        'object_id': object_id,
                        'cut_type': cut_type,
                        'variation': var_dir,
                        'partA': partA_path,
                        'partB': partB_path
                    })
        
        return pairs
    
    def _generate_test_samples(self):
        """テストサンプルを生成"""
        samples = []
        
        for obj_id in self.test_objects:
            pairs = self._get_mating_pairs(obj_id)
            
            if self.test_type == "binary":
                samples.extend(self._generate_binary_samples(pairs))
            elif self.test_type == "multi_choice":
                samples.extend(self._generate_multi_choice_samples(pairs))
                
        print(f"Generated {len(samples)} test samples for {self.test_type} task")
        return samples
    
    def _generate_binary_samples(self, pairs):
        """Binary mating samples生成"""
        samples = []
        
        for pair in pairs:
            # Positive sample (真のペア)
            samples.append({
                'type': 'positive',
                'pair_info': pair,
                'pc1_path': pair['partA'],
                'pc2_path': pair['partB'],
                'ground_truth': True
            })
            
            # Negative sample (異なるオブジェクトのパーツとのペア)
            other_objects = [obj for obj in self.test_objects if obj != pair['object_id']]
            if other_objects:
                other_obj = random.choice(other_objects)
                other_pairs = self._get_mating_pairs(other_obj)
                if other_pairs:
                    other_pair = random.choice(other_pairs)
                    samples.append({
                        'type': 'negative',
                        'pair_info': pair,
                        'pc1_path': pair['partA'],
                        'pc2_path': other_pair['partA'],  # 異なるオブジェクトのパーツ
                        'ground_truth': False
                    })
                    
        return samples
    
    def _generate_multi_choice_samples(self, pairs):
        """Multi-choice samples生成"""
        samples = []
        
        for pair in pairs:
            # 他のオブジェクトから2つのdistractorを選択
            other_objects = [obj for obj in self.test_objects if obj != pair['object_id']]
            if len(other_objects) >= 2:
                distractor_objects = random.sample(other_objects, 2)
                distractors = []
                
                for dist_obj in distractor_objects:
                    dist_pairs = self._get_mating_pairs(dist_obj)
                    if dist_pairs:
                        dist_pair = random.choice(dist_pairs)
                        distractors.append(dist_pair['partA'])
                
                if len(distractors) == 2:
                    # ランダムに配置（正解インデックスをランダム化）
                    correct_idx = random.randint(0, 2)
                    choices = [None, None, None]
                    choices[correct_idx] = pair['partB']  # 正解
                    
                    dist_idx = 0
                    for i in range(3):
                        if choices[i] is None:
                            choices[i] = distractors[dist_idx]
                            dist_idx += 1
                    
                    samples.append({
                        'type': 'multi_choice',
                        'pair_info': pair,
                        'query_pc': pair['partA'],
                        'choice_pcs': choices,
                        'ground_truth': correct_idx
                    })
                    
        return samples
    
    def __len__(self):
        return len(self.test_samples)
    
    def __getitem__(self, idx):
        sample = self.test_samples[idx]
        
        if sample['type'] in ['positive', 'negative']:
            # Binary task
            pc1 = self._load_point_cloud(sample['pc1_path'])
            pc2 = self._load_point_cloud(sample['pc2_path'])
            
            return {
                'type': 'binary',
                'point_clouds': torch.stack([pc1, pc2], dim=0),  # (2, N, C)
                'ground_truth': sample['ground_truth'],
                'pair_info': sample['pair_info'],
                'sample_idx': idx
            }
            
        elif sample['type'] == 'multi_choice':
            # Multi-choice task
            query_pc = self._load_point_cloud(sample['query_pc'])
            choice_pcs = [self._load_point_cloud(path) for path in sample['choice_pcs']]
            
            # query + 3 choices = 4 point clouds
            all_pcs = [query_pc] + choice_pcs
            
            return {
                'type': 'multi_choice',
                'point_clouds': torch.stack(all_pcs, dim=0),  # (4, N, C)
                'ground_truth': sample['ground_truth'],
                'pair_info': sample['pair_info'],
                'sample_idx': idx
            }

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:0")
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        use_cache=True, 
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    # 推論専用では PointNet を固定する
    model.get_model().fix_pointnet = True

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

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

def start_binary_evaluation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = BINARY_MATING_PROMPTS[prompt_index]

    results = {"task_type": "binary_mating", "prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    # Binary task用のプロンプト生成 (2つの点群)
    if mm_use_point_start_end:
        point_tokens = ""
        for i in range(2):  # Binary task: 2 point clouds
            point_tokens += default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + " "
        qs = point_tokens + qs
    else:
        raise ValueError("Multiple point clouds are not supported with point_patch mode. Use mm_use_point_start_end=True.")
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids_ = torch.as_tensor(inputs.input_ids).cuda()
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(dataloader):
        try:
            multi_point_clouds = batch["point_clouds"].squeeze(0).cuda().to(model.dtype)  # (2, N, C)
            ground_truth = batch["ground_truth"]
            pair_info = batch["pair_info"]
            sample_idx = batch["sample_idx"]

            input_ids = input_ids_.clone()
            multi_point_clouds_4d = multi_point_clouds.unsqueeze(0)  # (1, 2, N, C)
            
            outputs = generate_outputs(model, tokenizer, input_ids, multi_point_clouds_4d, stopping_criteria)
            
            # 出力をyes/no判定
            output_text = outputs[0].lower()
            predicted = "yes" in output_text or "true" in output_text or "fit" in output_text or "mate" in output_text
            
            is_correct = predicted == ground_truth.item()
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            result_dict = {
                "sample_id": sample_idx.item(),
                "ground_truth": ground_truth.item(),
                "predicted": predicted,
                "model_output": outputs[0],
                "is_correct": is_correct,
                "pair_info": pair_info
            }
            
            responses.append(result_dict)
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    results["results"] = responses
    results["accuracy"] = accuracy
    results["correct_predictions"] = correct_predictions
    results["total_predictions"] = total_predictions

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    print(f"Binary Mating Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def start_multi_choice_evaluation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = MULTI_CHOICE_PROMPTS[prompt_index]

    results = {"task_type": "multi_choice_mating", "prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    # Multi-choice task用のプロンプト生成 (4つの点群: query + 3 choices)
    if mm_use_point_start_end:
        point_tokens = ""
        for i in range(4):  # Multi-choice task: 4 point clouds
            point_tokens += default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + " "
        qs = point_tokens + qs
    else:
        raise ValueError("Multiple point clouds are not supported with point_patch mode. Use mm_use_point_start_end=True.")
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids_ = torch.as_tensor(inputs.input_ids).cuda()
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(dataloader):
        try:
            multi_point_clouds = batch["point_clouds"].squeeze(0).cuda().to(model.dtype)  # (4, N, C)
            ground_truth = batch["ground_truth"]
            pair_info = batch["pair_info"]
            sample_idx = batch["sample_idx"]

            input_ids = input_ids_.clone()
            multi_point_clouds_4d = multi_point_clouds.unsqueeze(0)  # (1, 4, N, C)
            
            outputs = generate_outputs(model, tokenizer, input_ids, multi_point_clouds_4d, stopping_criteria)
            
            # 出力から数字を抽出
            output_text = outputs[0].lower()
            predicted = -1
            for i in range(3):
                if str(i) in output_text:
                    predicted = i
                    break
            
            is_correct = predicted == ground_truth.item()
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            result_dict = {
                "sample_id": sample_idx.item(),
                "ground_truth": ground_truth.item(),
                "predicted": predicted,
                "model_output": outputs[0],
                "is_correct": is_correct,
                "pair_info": pair_info
            }
            
            responses.append(result_dict)
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    results["results"] = responses
    results["accuracy"] = accuracy
    results["correct_predictions"] = correct_predictions
    results["total_predictions"] = total_predictions

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    print(f"Multi-choice Mating Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    # 乱数シードを固定（再現性のため）
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model_basename = os.path.basename(os.path.expanduser(args.model_name))

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_name, "evaluation_orb3d")
        
    args.output_file = f"ORB3D_{args.test_type}_mating_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # 結果生成または読み込み
    if not os.path.exists(args.output_file_path):
        # データセット作成
        dataset = ORB3DShapeMatingDataset(
            data_path=args.data_path,
            npoints=args.npoints,
            test_type=args.test_type
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
        model, tokenizer, conv = init_model(args)

        print(f'[INFO] Start generating results for {args.output_file}.')
        
        if args.test_type == "binary":
            results = start_binary_evaluation(
                model, tokenizer, conv, dataloader, args.prompt_index, 
                args.output_dir, args.output_file
            )
        elif args.test_type == "multi_choice":
            results = start_multi_choice_evaluation(
                model, tokenizer, conv, dataloader, args.prompt_index, 
                args.output_dir, args.output_file
            )

        # メモリ解放
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    print(f"[INFO] ORB3D shape mating evaluation completed.")
    print(f"Task: {args.test_type}")
    print(f"Accuracy: {results.get('accuracy', 'N/A')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
        default="/groups/gag51402/datasets/ORB3D/objaverse_PWN_filtered/objaverse",
        help="Path to ORB3D dataset")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Output directory for results")
    parser.add_argument("--model_name", type=str, 
        default="RunsenXu/PointLLM_7B_v1.2")

    # dataset settings
    parser.add_argument("--npoints", type=int, default=8192, 
        help="Number of points to sample from point clouds")
    parser.add_argument("--test_type", type=str, choices=["binary", "multi_choice"], 
        default="binary", help="Type of mating test")

    # data loader settings
    parser.add_argument("--batch_size", type=int, default=1, help="Keep batch_size=1")
    parser.add_argument("--num_workers", type=int, default=0, help="Use 0 to avoid memory issues")

    # task specific settings
    parser.add_argument("--prompt_index", type=int, default=0, 
        help="Index of prompt in respective prompt list")

    args = parser.parse_args()

    main(args) 
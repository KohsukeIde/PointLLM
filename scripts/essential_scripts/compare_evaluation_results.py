#!/usr/bin/env python3

import json
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_evaluation_results(file_path):
    """評価結果JSONを読み込み"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_ordinal_accuracy(results):
    """序数参照精度を分析"""
    if not results or 'results' not in results:
        return {}
    
    analysis = {
        'total_samples': len(results['results']),
        'exact_matches': 0,
        'semantic_matches': 0,
        'by_objects': defaultdict(lambda: {'exact': 0, 'semantic': 0, 'total': 0}),
        'by_question_type': defaultdict(lambda: {'exact': 0, 'semantic': 0, 'total': 0})
    }
    
    for result in results['results']:
        if 'accuracy' in result and isinstance(result['accuracy'], dict):
            acc = result['accuracy']
            exact = acc.get('exact_match', 0)
            semantic = acc.get('semantic_match', 0)
            
            analysis['exact_matches'] += exact
            analysis['semantic_matches'] += semantic
            
            # オブジェクト数別統計
            num_objects = result.get('num_objects', 0)
            analysis['by_objects'][num_objects]['exact'] += exact
            analysis['by_objects'][num_objects]['semantic'] += semantic
            analysis['by_objects'][num_objects]['total'] += 1
            
            # 質問タイプ別統計
            question = result.get('question', '').lower()
            if 'first object' in question:
                qtype = 'first_object'
            elif 'second object' in question:
                qtype = 'second_object'
            elif 'third object' in question:
                qtype = 'third_object'
            elif 'same kind' in question:
                qtype = 'same_kind'
            elif 'different' in question:
                qtype = 'different'
            elif 'compare' in question:
                qtype = 'compare'
            else:
                qtype = 'other'
            
            analysis['by_question_type'][qtype]['exact'] += exact
            analysis['by_question_type'][qtype]['semantic'] += semantic
            analysis['by_question_type'][qtype]['total'] += 1
    
    # 精度を計算
    total = analysis['total_samples']
    if total > 0:
        analysis['exact_accuracy'] = analysis['exact_matches'] / total
        analysis['semantic_accuracy'] = analysis['semantic_matches'] / total
    else:
        analysis['exact_accuracy'] = 0
        analysis['semantic_accuracy'] = 0
    
    return analysis

def analyze_multi_object_results(results):
    """マルチオブジェクト評価結果を分析"""
    if not results or 'results' not in results:
        return {}
    
    analysis = {
        'total_samples': len(results['results']),
        'by_objects': defaultdict(lambda: {'count': 0, 'responses': []}),
        'response_lengths': [],
        'category_distribution': defaultdict(int)
    }
    
    for result in results['results']:
        num_objects = result.get('num_objects', 0)
        model_output = result.get('model_output', '')
        object_details = result.get('object_details', [])
        
        analysis['by_objects'][num_objects]['count'] += 1
        analysis['by_objects'][num_objects]['responses'].append(len(model_output))
        analysis['response_lengths'].append(len(model_output))
        
        # カテゴリ分布
        for detail in object_details:
            category = detail.get('category', 'unknown')
            analysis['category_distribution'][category] += 1
    
    return analysis

def compare_results(base_results, lora_results, output_dir):
    """ベースモデルとLoRAモデルの結果を比較"""
    print("=" * 60)
    print("Evaluation Results Comparison")
    print("=" * 60)
    
    # 基本統計
    print("\n1. Basic Statistics:")
    if base_results and 'results' in base_results:
        print(f"   Base Model Samples: {len(base_results['results'])}")
    if lora_results and 'results' in lora_results:
        print(f"   LoRA Model Samples: {len(lora_results['results'])}")
    
    # 序数参照精度の比較
    if base_results and lora_results:
        base_analysis = analyze_ordinal_accuracy(base_results)
        lora_analysis = analyze_ordinal_accuracy(lora_results)
        
        print("\n2. Ordinal Reference Accuracy Comparison:")
        print(f"   Base Model - Exact: {base_analysis['exact_accuracy']:.3f}, Semantic: {base_analysis['semantic_accuracy']:.3f}")
        print(f"   LoRA Model - Exact: {lora_analysis['exact_accuracy']:.3f}, Semantic: {lora_analysis['semantic_accuracy']:.3f}")
        
        # 改善率
        exact_improvement = lora_analysis['exact_accuracy'] - base_analysis['exact_accuracy']
        semantic_improvement = lora_analysis['semantic_accuracy'] - base_analysis['semantic_accuracy']
        print(f"   Improvement - Exact: {exact_improvement:+.3f}, Semantic: {semantic_improvement:+.3f}")
        
        # オブジェクト数別比較
        print("\n3. Accuracy by Number of Objects:")
        for num_objects in sorted(set(base_analysis['by_objects'].keys()) | set(lora_analysis['by_objects'].keys())):
            base_obj = base_analysis['by_objects'][num_objects]
            lora_obj = lora_analysis['by_objects'][num_objects]
            
            base_exact = base_obj['exact'] / max(base_obj['total'], 1)
            base_semantic = base_obj['semantic'] / max(base_obj['total'], 1)
            lora_exact = lora_obj['exact'] / max(lora_obj['total'], 1)
            lora_semantic = lora_obj['semantic'] / max(lora_obj['total'], 1)
            
            print(f"   {num_objects} objects:")
            print(f"     Base: Exact={base_exact:.3f}, Semantic={base_semantic:.3f}")
            print(f"     LoRA: Exact={lora_exact:.3f}, Semantic={lora_semantic:.3f}")
            print(f"     Δ: Exact={lora_exact-base_exact:+.3f}, Semantic={lora_semantic-base_semantic:+.3f}")
        
        # 質問タイプ別比較
        print("\n4. Accuracy by Question Type:")
        for qtype in sorted(set(base_analysis['by_question_type'].keys()) | set(lora_analysis['by_question_type'].keys())):
            base_q = base_analysis['by_question_type'][qtype]
            lora_q = lora_analysis['by_question_type'][qtype]
            
            base_exact = base_q['exact'] / max(base_q['total'], 1)
            lora_exact = lora_q['exact'] / max(lora_q['total'], 1)
            
            print(f"   {qtype}: Base={base_exact:.3f}, LoRA={lora_exact:.3f}, Δ={lora_exact-base_exact:+.3f}")
        
        # グラフ作成
        create_comparison_plots(base_analysis, lora_analysis, output_dir)

def create_comparison_plots(base_analysis, lora_analysis, output_dir):
    """比較グラフを作成"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 全体精度比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Exact Match
    models = ['Base Model', 'LoRA Model']
    exact_scores = [base_analysis['exact_accuracy'], lora_analysis['exact_accuracy']]
    semantic_scores = [base_analysis['semantic_accuracy'], lora_analysis['semantic_accuracy']]
    
    ax1.bar(models, exact_scores, alpha=0.7, color=['blue', 'red'])
    ax1.set_title('Exact Match Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for i, score in enumerate(exact_scores):
        ax1.text(i, score + 0.01, f'{score:.3f}', ha='center')
    
    # Semantic Match
    ax2.bar(models, semantic_scores, alpha=0.7, color=['blue', 'red'])
    ax2.set_title('Semantic Match Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    for i, score in enumerate(semantic_scores):
        ax2.text(i, score + 0.01, f'{score:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # オブジェクト数別比較
    if base_analysis['by_objects'] and lora_analysis['by_objects']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        object_nums = sorted(set(base_analysis['by_objects'].keys()) | set(lora_analysis['by_objects'].keys()))
        x = np.arange(len(object_nums))
        width = 0.35
        
        base_scores = []
        lora_scores = []
        
        for num in object_nums:
            base_obj = base_analysis['by_objects'][num]
            lora_obj = lora_analysis['by_objects'][num]
            
            base_score = base_obj['exact'] / max(base_obj['total'], 1)
            lora_score = lora_obj['exact'] / max(lora_obj['total'], 1)
            
            base_scores.append(base_score)
            lora_scores.append(lora_score)
        
        ax.bar(x - width/2, base_scores, width, label='Base Model', alpha=0.7)
        ax.bar(x + width/2, lora_scores, width, label='LoRA Model', alpha=0.7)
        
        ax.set_xlabel('Number of Objects')
        ax.set_ylabel('Exact Match Accuracy')
        ax.set_title('Accuracy by Number of Objects')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{num} objects' for num in object_nums])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_objects.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--base_results", type=str, 
                        default="evaluation/baseline/ModelNet_multi_classification_prompt2_obj3.json",
                        help="Path to base model results")
    parser.add_argument("--lora_results", type=str,
                        default="evaluation/lora_ordinal/ordinal_evaluation_modedataset.json",
                        help="Path to LoRA model results")
    parser.add_argument("--output_dir", type=str, default="evaluation/comparison",
                        help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # 結果読み込み
    print("Loading evaluation results...")
    base_results = load_evaluation_results(args.base_results)
    lora_results = load_evaluation_results(args.lora_results)
    
    # 比較実行
    compare_results(base_results, lora_results, args.output_dir)
    
    # マルチオブジェクト結果の詳細分析
    if base_results:
        base_multi_analysis = analyze_multi_object_results(base_results)
        print(f"\nBase Model Multi-Object Analysis:")
        print(f"  Total samples: {base_multi_analysis['total_samples']}")
        for num_objects, data in base_multi_analysis['by_objects'].items():
            avg_response_len = np.mean(data['responses']) if data['responses'] else 0
            print(f"  {num_objects} objects: {data['count']} samples, avg response length: {avg_response_len:.1f}")

if __name__ == "__main__":
    main() 
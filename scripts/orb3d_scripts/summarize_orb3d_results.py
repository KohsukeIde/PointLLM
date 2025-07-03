#!/usr/bin/env python3
# scripts/summarize_orb3d_results.py

import argparse
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_results(results_dir):
    """結果ファイルを読み込み"""
    binary_results = []
    multi_choice_results = []
    
    # Binaryタスクの結果
    binary_files = glob.glob(os.path.join(results_dir, "ORB3D_binary_mating_prompt*.json"))
    for file_path in sorted(binary_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            prompt_idx = int(os.path.basename(file_path).split('prompt')[1].split('.')[0])
            binary_results.append({
                'prompt_index': prompt_idx,
                'prompt': data['prompt'],
                'accuracy': data['accuracy'],
                'correct': data['correct_predictions'],
                'total': data['total_predictions']
            })
    
    # Multi-choiceタスクの結果
    multi_files = glob.glob(os.path.join(results_dir, "ORB3D_multi_choice_mating_prompt*.json"))
    for file_path in sorted(multi_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            prompt_idx = int(os.path.basename(file_path).split('prompt')[1].split('.')[0])
            multi_choice_results.append({
                'prompt_index': prompt_idx,
                'prompt': data['prompt'],
                'accuracy': data['accuracy'],
                'correct': data['correct_predictions'],
                'total': data['total_predictions']
            })
    
    return binary_results, multi_choice_results

def analyze_results_by_cut_type(results_dir):
    """切り口タイプ別の性能分析"""
    binary_files = glob.glob(os.path.join(results_dir, "ORB3D_binary_mating_prompt*.json"))
    multi_files = glob.glob(os.path.join(results_dir, "ORB3D_multi_choice_mating_prompt*.json"))
    
    cut_type_analysis = {
        'binary': {'planar': [], 'parabolic': []},
        'multi_choice': {'planar': [], 'parabolic': []}
    }
    
    for file_path in binary_files + multi_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            task_type = 'binary' if 'binary' in file_path else 'multi_choice'
            
            for result in data['results']:
                cut_type = result['pair_info']['cut_type']
                is_correct = result['is_correct']
                cut_type_analysis[task_type][cut_type].append(is_correct)
    
    # 統計計算
    statistics = {}
    for task_type in ['binary', 'multi_choice']:
        statistics[task_type] = {}
        for cut_type in ['planar', 'parabolic']:
            if cut_type_analysis[task_type][cut_type]:
                correct_count = sum(cut_type_analysis[task_type][cut_type])
                total_count = len(cut_type_analysis[task_type][cut_type])
                accuracy = correct_count / total_count
                statistics[task_type][cut_type] = {
                    'accuracy': accuracy,
                    'correct': correct_count,
                    'total': total_count
                }
            else:
                statistics[task_type][cut_type] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }
    
    return statistics

def create_summary_plots(binary_results, multi_choice_results, cut_type_stats, output_dir):
    """要約プロットを作成"""
    
    # Plot 1: Prompt別のAccuracy比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Binary task
    prompt_indices = [r['prompt_index'] for r in binary_results]
    binary_accuracies = [r['accuracy'] for r in binary_results]
    ax1.bar(prompt_indices, binary_accuracies, alpha=0.7, color='blue')
    ax1.set_title('Binary Mating Task Accuracy by Prompt')
    ax1.set_xlabel('Prompt Index')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Multi-choice task
    prompt_indices = [r['prompt_index'] for r in multi_choice_results]
    multi_accuracies = [r['accuracy'] for r in multi_choice_results]
    ax2.bar(prompt_indices, multi_accuracies, alpha=0.7, color='green')
    ax2.set_title('Multi-choice Mating Task Accuracy by Prompt')
    ax2.set_xlabel('Prompt Index')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_prompt.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: 切り口タイプ別の性能比較
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    cut_types = ['planar', 'parabolic']
    binary_accs = [cut_type_stats['binary'][ct]['accuracy'] for ct in cut_types]
    multi_accs = [cut_type_stats['multi_choice'][ct]['accuracy'] for ct in cut_types]
    
    x = np.arange(len(cut_types))
    width = 0.35
    
    ax.bar(x - width/2, binary_accs, width, label='Binary Task', alpha=0.7, color='blue')
    ax.bar(x + width/2, multi_accs, width, label='Multi-choice Task', alpha=0.7, color='green')
    
    ax.set_xlabel('Cut Type')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Cut Type')
    ax.set_xticks(x)
    ax.set_xticklabels(cut_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_cut_type.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(binary_results, multi_choice_results, cut_type_stats, output_dir):
    """テキスト要約レポートを生成"""
    
    report_path = os.path.join(output_dir, 'orb3d_mating_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("ORB3D Shape Mating Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("Overall Results:\n")
        f.write("-" * 20 + "\n")
        
        if binary_results:
            avg_binary_acc = np.mean([r['accuracy'] for r in binary_results])
            total_binary_samples = sum([r['total'] for r in binary_results])
            f.write(f"Binary Task Average Accuracy: {avg_binary_acc:.4f}\n")
            f.write(f"Binary Task Total Samples: {total_binary_samples}\n")
        
        if multi_choice_results:
            avg_multi_acc = np.mean([r['accuracy'] for r in multi_choice_results])
            total_multi_samples = sum([r['total'] for r in multi_choice_results])
            f.write(f"Multi-choice Task Average Accuracy: {avg_multi_acc:.4f}\n")
            f.write(f"Multi-choice Task Total Samples: {total_multi_samples}\n")
        
        f.write("\n")
        
        # Prompt-wise results
        f.write("Binary Task Results by Prompt:\n")
        f.write("-" * 30 + "\n")
        for result in binary_results:
            f.write(f"Prompt {result['prompt_index']}: {result['accuracy']:.4f} "
                   f"({result['correct']}/{result['total']})\n")
            f.write(f"  \"{result['prompt']}\"\n")
        
        f.write("\nMulti-choice Task Results by Prompt:\n")
        f.write("-" * 35 + "\n")
        for result in multi_choice_results:
            f.write(f"Prompt {result['prompt_index']}: {result['accuracy']:.4f} "
                   f"({result['correct']}/{result['total']})\n")
            f.write(f"  \"{result['prompt']}\"\n")
        
        # Cut type analysis
        f.write("\nCut Type Analysis:\n")
        f.write("-" * 20 + "\n")
        for task_type in ['binary', 'multi_choice']:
            f.write(f"\n{task_type.title()} Task:\n")
            for cut_type in ['planar', 'parabolic']:
                stats = cut_type_stats[task_type][cut_type]
                f.write(f"  {cut_type}: {stats['accuracy']:.4f} "
                       f"({stats['correct']}/{stats['total']})\n")
    
    print(f"Summary report saved to: {report_path}")

def main(args):
    results_dir = args.results_dir
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print("Loading results...")
    binary_results, multi_choice_results = load_results(results_dir)
    
    print("Analyzing results by cut type...")
    cut_type_stats = analyze_results_by_cut_type(results_dir)
    
    print("Creating summary plots...")
    create_summary_plots(binary_results, multi_choice_results, cut_type_stats, results_dir)
    
    print("Generating summary report...")
    generate_summary_report(binary_results, multi_choice_results, cut_type_stats, results_dir)
    
    print("Summary generation completed!")
    
    # Display quick summary
    if binary_results:
        avg_binary = np.mean([r['accuracy'] for r in binary_results])
        print(f"\nBinary Task Average Accuracy: {avg_binary:.4f}")
    
    if multi_choice_results:
        avg_multi = np.mean([r['accuracy'] for r in multi_choice_results])
        print(f"Multi-choice Task Average Accuracy: {avg_multi:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize ORB3D Shape Mating evaluation results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing evaluation result JSON files")
    
    args = parser.parse_args()
    main(args) 
#!/usr/bin/env python3
"""
PointLLM Self-Attention Weight Analysis Tool
Grad-CAM風の可視化でモデルがどの点群ブロックに注意を向けているかを分析
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("[WARNING] Seaborn not available, using matplotlib only")
from matplotlib.patches import Rectangle
import os
import json
from tqdm import tqdm
from collections import defaultdict

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import ModelNet
from transformers import AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class AttentionHook:
    """Self-attentionの重みをフックして取得するクラス"""
    
    def __init__(self):
        self.attention_weights = []
        self.token_ids = None
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        """Attention重みを保存するフック関数"""
        # output[1]がattention weights: (batch_size, num_heads, seq_len, seq_len)
        if len(output) > 1 and output[1] is not None:
            attn_weights = output[1].detach().cpu()
            self.attention_weights.append(attn_weights)
    
    def register_hooks(self, model):
        """モデルの全てのattentionレイヤーにフックを登録"""
        self.attention_weights = []
        
        # Transformerのattentionモジュールを見つけてフックを登録
        for name, module in model.named_modules():
            if 'self_attn' in name or 'attention' in name:
                if hasattr(module, 'forward') and 'attn' in name.lower():
                    hook = module.register_forward_hook(self.hook_fn)
                    self.hooks.append(hook)
                    print(f"[DEBUG] Registered hook for: {name}")
    
    def remove_hooks(self):
        """登録したフックを削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_averaged_attention(self):
        """全レイヤーの注意重みを平均化"""
        if not self.attention_weights:
            return None
        
        # 最後のいくつかのレイヤーを使用（より重要な特徴を捉える）
        last_layers = self.attention_weights[-4:]  # 最後の4レイヤー
        
        # 平均を計算: (batch_size, num_heads, seq_len, seq_len)
        avg_attention = torch.mean(torch.stack(last_layers), dim=0)
        
        # ヘッド方向も平均化: (batch_size, seq_len, seq_len)
        avg_attention = torch.mean(avg_attention, dim=1)
        
        return avg_attention

def load_model_with_lora(model_name, lora_dir=None):
    """LoRAを含むモデルを読み込み"""
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    
    print(f'[INFO] Base model name: {os.path.basename(model_name)}')
    
    # トークナイザー読み込み
    if lora_dir and os.path.isdir(lora_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(lora_dir)
            print(f"[INFO] Loaded tokenizer from LoRA directory: {lora_dir}")
        except Exception as e:
            print(f"Failed to load tokenizer from LoRA directory: {e}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda:0")
    
    # ベースモデル読み込み（フォールバック処理追加）
    try:
        base_model = PointLLMLlamaForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            use_cache=True,
            torch_dtype=torch.bfloat16,
            output_attentions=True,  # attention重みを出力
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
                        output_attentions=True,
                        trust_remote_code=True
                    ).to(device)
                    print(f"Successfully loaded model from: {fallback_path}")
                    break
                except Exception as e_fallback:
                    print(f"Failed to load from {fallback_path}: {e_fallback}")
                    continue
        else:
            raise ValueError("Could not load PointLLM model from any available path")
    
    # LoRA読み込み
    if lora_dir and PEFT_AVAILABLE:
        print(f"[INFO] Loading LoRA from: {lora_dir}")
        try:
            model = PeftModel.from_pretrained(base_model, lora_dir)
            print('[INFO] LoRA adapter loaded successfully')
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print('[WARNING] Size mismatch detected, loading with ignore_mismatched_sizes=True')
                model = PeftModel.from_pretrained(base_model, lora_dir, ignore_mismatched_sizes=True)
                print('[INFO] LoRA adapter loaded with size mismatch ignored')
            else:
                raise e
    else:
        model = base_model
    
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    # カスタムトークンの埋め込み行列拡張
    if lora_dir:
        custom_tokens_file = os.path.join(lora_dir, 'custom_tokens.json')
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
    
    model.get_model().fix_pointnet = True
    return model, tokenizer

def analyze_attention_weights(attention_weights, token_ids, tokenizer, point_backbone_config):
    """Attention重みを分析してpoint cloudブロックへの注意度を計算"""
    batch_size, seq_len, _ = attention_weights.shape
    
    results = []
    
    for batch_idx in range(batch_size):
        tokens = token_ids[batch_idx]
        attn_matrix = attention_weights[batch_idx]  # (seq_len, seq_len)
        
        # トークンIDを文字列に変換
        token_strings = [tokenizer.decode([tid.item()]) for tid in tokens]
        
        # point cloudブロックの位置を特定
        point_start_token = point_backbone_config['point_start_token']
        point_end_token = point_backbone_config['point_end_token']
        
        point_start_positions = torch.where(tokens == point_start_token)[0].tolist()
        point_end_positions = torch.where(tokens == point_end_token)[0].tolist()
        
        # 識別トークンの位置を特定
        pc_tokens = ['<pcA>', '<pcB>', '<pcC>']
        pc_positions = []
        for pc_token in pc_tokens:
            pc_id = tokenizer.convert_tokens_to_ids(pc_token)
            if pc_id != tokenizer.unk_token_id:
                positions = torch.where(tokens == pc_id)[0].tolist()
                pc_positions.extend([(pos, pc_token) for pos in positions])
        
        # 点群ブロックごとの注意度を計算
        point_cloud_blocks = []
        for i, (start_pos, end_pos) in enumerate(zip(point_start_positions, point_end_positions)):
            # ブロック全体への注意度を計算（最後のトークンからの注意）
            last_token_attention = attn_matrix[-1, start_pos:end_pos+1]  # 最後のトークンから各点群ブロックへ
            avg_attention_to_block = last_token_attention.mean().item()
            
            # 該当する識別トークンを探す
            pc_token = None
            for pos, token in pc_positions:
                if abs(pos - start_pos) <= 2:  # 近い位置にある識別トークン
                    pc_token = token
                    break
            
            point_cloud_blocks.append({
                'block_index': i,
                'pc_token': pc_token or f'pc{i}',
                'start_pos': start_pos,
                'end_pos': end_pos,
                'block_length': end_pos - start_pos + 1,
                'attention_weight': avg_attention_to_block
            })
        
        # 全体の統計
        total_attention = sum(block['attention_weight'] for block in point_cloud_blocks)
        for block in point_cloud_blocks:
            block['attention_percentage'] = (block['attention_weight'] / total_attention) * 100 if total_attention > 0 else 0
        
        results.append({
            'batch_index': batch_idx,
            'point_cloud_blocks': point_cloud_blocks,
            'total_attention': total_attention,
            'sequence_length': seq_len,
            'num_point_blocks': len(point_cloud_blocks)
        })
    
    return results

def visualize_attention_heatmap(attention_weights, token_ids, tokenizer, point_backbone_config, save_path=None):
    """Attention重みをヒートマップで可視化"""
    batch_size = attention_weights.shape[0]
    
    fig, axes = plt.subplots(1, batch_size, figsize=(6*batch_size, 8))
    if batch_size == 1:
        axes = [axes]
    
    for batch_idx in range(batch_size):
        ax = axes[batch_idx]
        tokens = token_ids[batch_idx]
        attn_matrix = attention_weights[batch_idx]
        
        # トークン文字列
        token_strings = [tokenizer.decode([tid.item()]) for tid in tokens]
        
        # 点群ブロックの位置
        point_start_token = point_backbone_config['point_start_token']
        point_end_token = point_backbone_config['point_end_token']
        point_start_positions = torch.where(tokens == point_start_token)[0].tolist()
        point_end_positions = torch.where(tokens == point_end_token)[0].tolist()
        
        # ヒートマップ描画
        if SEABORN_AVAILABLE:
            sns.heatmap(attn_matrix.float().numpy(), ax=ax, cmap='Blues', cbar=True, square=True)
        else:
            # matplotlibのみを使用したヒートマップ
            im = ax.imshow(attn_matrix.float().numpy(), cmap='Blues', aspect='auto')
            plt.colorbar(im, ax=ax)
        
        # 点群ブロックをハイライト
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, (start_pos, end_pos) in enumerate(zip(point_start_positions, point_end_positions)):
            color = colors[i % len(colors)]
            # 縦の線
            ax.axvline(x=start_pos, color=color, linewidth=2, alpha=0.7)
            ax.axvline(x=end_pos+1, color=color, linewidth=2, alpha=0.7)
            # 横の線
            ax.axhline(y=start_pos, color=color, linewidth=2, alpha=0.7)
            ax.axhline(y=end_pos+1, color=color, linewidth=2, alpha=0.7)
            
            # ラベル
            ax.text(start_pos + (end_pos-start_pos)/2, -2, f'PC{i+1}', 
                   ha='center', va='top', color=color, fontweight='bold')
        
        ax.set_title(f'Attention Weights (Batch {batch_idx})')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Attention heatmap saved to: {save_path}")
    
    return fig

def create_positional_bias_plot(analysis_results, save_path=None):
    """位置バイアス分析の単体プロット（理想分布オーバーレイ付き）"""
    # プロフェッショナルな学術的色彩パレット（FAIR論文風）
    fair_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # 青、紫、オレンジ、赤
    
    # 学術論文風のスタイル設定
    plt.style.use('default')  # リセット
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'text.usetex': False,  # LaTeXは使わない（依存関係を避ける）
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.bottom': True,
        'ytick.left': True,
        'figure.facecolor': 'white'
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 位置による注意度バイアス
    position_bias = defaultdict(list)
    for result in analysis_results:
        for block in result['point_cloud_blocks']:
            position_bias[block['block_index']].append(block['attention_percentage'])
    
    positions = sorted(position_bias.keys())
    avg_attentions = [np.mean(position_bias[pos]) for pos in positions]
    std_attentions = [np.std(position_bias[pos]) for pos in positions]
    
    # 実際の分布をバープロット（凡例なし）
    bars = []
    for i, (pos, avg, std) in enumerate(zip(positions, avg_attentions, std_attentions)):
        bar = ax.bar(pos, avg, yerr=std, capsize=8, 
                     color=fair_colors[i], alpha=0.8, 
                     edgecolor='white', linewidth=1.5, error_kw={'linewidth': 2})
        bars.append(bar)
    
    # Statistical RangeとIdeal線を削除（シンプルな表示）
    
    ax.set_xlabel('Point Cloud Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Attention Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Positional Bias in Attention Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Position {i+1}' for i in positions], fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 凡例を削除（シンプルな表示）
    
    # 数値ラベル（各バーに個別に）
    for i, (pos, avg, std) in enumerate(zip(positions, avg_attentions, std_attentions)):
        ax.text(pos, avg + std + max(avg_attentions)*0.02, 
                f'{avg:.1f}%\n±{std:.1f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # バイアス強度の判定と表示（シンプル版）
    if len(positions) >= 2:
        last_position_attention = avg_attentions[-1]
        
        # シンプルな閾値判定
        if last_position_attention > 60:
            bias_level = "HIGH BIAS"
            text_color = '#cc0000'  # 濃い赤
        elif last_position_attention > 45:
            bias_level = "MODERATE BIAS"
            text_color = '#cc8800'  # オレンジ
        else:
            bias_level = "LOW BIAS"
            text_color = '#008800'  # 緑
        
        # 左上にバイアスレベルを表示
        textstr = f'Last Position Bias Level: {bias_level}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', horizontalalignment='left', 
                color=text_color, fontweight='bold')
    
    # スタイル調整
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[INFO] Positional bias plot saved to: {save_path}")
    
    return fig

def create_attention_summary_plot(analysis_results, save_path=None):
    """注意度の要約プロットを作成（Kaiming He FAIR スタイル）"""
    # プロフェッショナルな学術的色彩パレット（FAIR論文風）
    fair_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # 青、紫、オレンジ、赤
    
    # 学術論文風のスタイル設定
    plt.style.use('default')  # リセット
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'text.usetex': False,  # LaTeXは使わない（依存関係を避ける）
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.bottom': True,
        'ytick.left': True,
        'figure.facecolor': 'white'
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 点群ブロック別の注意度分布
    all_attentions = []
    block_labels = []
    for result in analysis_results:
        for block in result['point_cloud_blocks']:
            all_attentions.append(block['attention_percentage'])
            block_labels.append(f"{block['pc_token']}")  # シンプルなラベル
    
    if all_attentions:  # データがある場合のみプロット
        # プロフェッショナルなバープロット
        bars = ax1.bar(range(len(all_attentions)), all_attentions, 
                       color=fair_colors[:len(all_attentions)], alpha=0.8, edgecolor='white', linewidth=1.5)
        ax1.set_xlabel('Point Cloud Blocks', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Attention Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Attention Distribution Across Point Cloud Blocks', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(range(len(block_labels)))
        ax1.set_xticklabels(block_labels, fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(all_attentions) * 1.15)
    else:
        ax1.text(0.5, 0.5, 'No attention data available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
        ax1.set_xlabel('Point Cloud Blocks', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Attention Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Attention Distribution Across Point Cloud Blocks', fontsize=14, fontweight='bold', pad=20)
    
        # 数値ラベル（よりクリーンなスタイル）
        for bar, attention in zip(bars, all_attentions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(all_attentions)*0.02,
                    f'{attention:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 位置による注意度バイアス（解釈ガイド付き）
    position_bias = defaultdict(list)
    for result in analysis_results:
        for block in result['point_cloud_blocks']:
            position_bias[block['block_index']].append(block['attention_percentage'])
    
    positions = sorted(position_bias.keys())
    avg_attentions = [np.mean(position_bias[pos]) for pos in positions]
    std_attentions = [np.std(position_bias[pos]) for pos in positions]
    
    bars2 = ax2.bar(positions, avg_attentions, yerr=std_attentions, capsize=8, 
                    color=fair_colors[:len(positions)], alpha=0.8, 
                    edgecolor='white', linewidth=1.5, error_kw={'linewidth': 2})
    ax2.set_xlabel('Point Cloud Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Attention Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Positional Bias Analysis', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f'Position {i+1}' for i in positions], fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 数値ラベル
    for i, (pos, avg, std) in enumerate(zip(positions, avg_attentions, std_attentions)):
        ax2.text(pos, avg + std + max(avg_attentions)*0.02, 
                f'{avg:.1f}%\n±{std:.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 解釈ガイドを追加
    if len(positions) >= 2:
        last_position_attention = avg_attentions[-1]
        
        # バイアス強度の判定
        if last_position_attention > 70:
            bias_level = "HIGH BIAS"
            box_color = '#ffcccc'  # 薄い赤
            text_color = '#cc0000'  # 濃い赤
        elif last_position_attention > 50:
            bias_level = "MODERATE BIAS"
            box_color = '#fff4cc'  # 薄い黄
            text_color = '#cc8800'  # オレンジ
        else:
            bias_level = "LOW BIAS"
            box_color = '#ccffcc'  # 薄い緑
            text_color = '#008800'  # 緑
        
        # 統計情報ボックス
        textstr = f'Last Position: {last_position_attention:.1f}%\nBias Level: {bias_level}'
        props = dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.8, edgecolor=text_color)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11, 
                verticalalignment='top', bbox=props, color=text_color, fontweight='bold')
        
        # 統計的解釈ガイドを図の下に追加
        expected_uniform = 100.0 / len(positions)
        guide_text = (
            "Statistical Interpretation Guide:\n"
            f"• Uniform distribution: ~{expected_uniform:.1f}% for each position ({len(positions)} objects)\n"
            f"• High bias (>2σ): Statistically significant positional preference\n"
            f"• Moderate bias (1-2σ): Some preference within expected variation\n"
            f"• Low bias (<1σ): Good balance, within normal statistical range\n"
            f"• σ (standard deviation) calculated from actual attention distribution"
        )
        
        fig.text(0.5, 0.02, guide_text, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.7),
                style='italic')
    
    # スタイル調整
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 解釈ガイド用のスペース確保
    
    # タイトルとサブタイトルを追加
    fig.suptitle('PointLLM Attention Analysis Results', fontsize=16, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[INFO] Attention summary (improved style) saved to: {save_path}")
    
    return fig

def run_attention_analysis(model, tokenizer, dataloader, output_dir, num_samples=10):
    """注意度分析を実行（プロンプト別期待値分析）"""
    model.eval()
    
    # Attention hookを設定
    attention_hook = AttentionHook()
    attention_hook.register_hooks(model)
    
    point_backbone_config = model.get_model().point_backbone_config
    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    
    # 複数プロンプトでの分析（期待値が異なる）
    test_prompts = [
        {"query": "What is the first object?", "expected_position": 0, "expected_weights": [0.7, 0.2, 0.1]},
        {"query": "What is the second object?", "expected_position": 1, "expected_weights": [0.15, 0.7, 0.15]},
        {"query": "What is the third object?", "expected_position": 2, "expected_weights": [0.1, 0.2, 0.7]},
        {"query": "Compare all objects.", "expected_position": -1, "expected_weights": [0.33, 0.34, 0.33]},  # 均等
    ]
    
    all_analysis_results = []
    
    try:
        with torch.no_grad():
            for sample_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing attention")):
                if sample_idx >= num_samples:
                    break
                
                # 複数プロンプトで分析
                for prompt_info in test_prompts:
                    query = prompt_info["query"]
                    expected_pos = prompt_info["expected_position"]
                    expected_weights = prompt_info["expected_weights"]
                    
                    # 点群トークン挿入
                    point_token_len = point_backbone_config['point_token_len']
                    default_point_patch_token = point_backbone_config['default_point_patch_token']
                    default_point_start_token = point_backbone_config['default_point_start_token']
                    default_point_end_token = point_backbone_config['default_point_end_token']
                    
                    tokens_with_id = ""
                    position_tokens = ["<pcA>", "<pcB>", "<pcC>"]
                    num_objects = batch["point_clouds"].shape[1] if len(batch["point_clouds"].shape) > 3 else 3
                    
                    for i in range(num_objects):
                        id_token = position_tokens[i] if i < len(position_tokens) else f"<pc{chr(ord('A') + i)}>"
                        tokens_with_id += (
                            id_token + " " +
                            default_point_start_token +
                            default_point_patch_token * point_token_len +
                            default_point_end_token + " "
                        )
                    
                    full_prompt = tokens_with_id + query
                
                conv.append_message(conv.roles[0], full_prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                inputs = tokenizer([prompt])
                input_ids = torch.as_tensor(inputs.input_ids).cuda()
                
                # 点群データ準備
                point_clouds = batch["point_clouds"].squeeze(0).cuda().to(model.dtype)
                if len(point_clouds.shape) == 2:  # (N, C)
                    point_clouds = point_clouds.unsqueeze(0)  # (1, N, C)
                if len(point_clouds.shape) == 3:  # (M, N, C)
                    point_clouds = point_clouds.unsqueeze(0)  # (1, M, N, C)
                
                # フォワードパス（注意重みを取得）
                attention_hook.attention_weights = []  # リセット
                
                try:
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids,
                            point_clouds=point_clouds,
                            max_length=input_ids.shape[1] + 50,
                            do_sample=False,
                            output_attentions=True,
                            return_dict_in_generate=True
                        )
                except Exception as e:
                    print(f"[WARNING] Generation failed for sample {sample_idx}: {e}")
                    continue
                
                # 注意重みを取得
                averaged_attention = attention_hook.get_averaged_attention()
                if averaged_attention is None:
                    print(f"[WARNING] No attention weights captured for sample {sample_idx}")
                    continue
                
                # 分析実行（期待値情報を追加）
                analysis_result = analyze_attention_weights(
                    averaged_attention, input_ids, tokenizer, point_backbone_config
                )
                
                # 期待値情報を各結果に追加
                for result in analysis_result:
                    result['prompt_info'] = prompt_info
                    result['query'] = query
                    result['expected_position'] = expected_pos
                    result['expected_weights'] = expected_weights
                
                all_analysis_results.extend(analysis_result)
                
                # 個別のヒートマップ保存（最初の数サンプルのみ）
                if sample_idx < 3:
                    heatmap_path = os.path.join(output_dir, f'attention_heatmap_sample_{sample_idx}.png')
                    visualize_attention_heatmap(
                        averaged_attention, input_ids, tokenizer, point_backbone_config, heatmap_path
                    )
                
                # conversationをリセット
                conv.messages = []
                
    finally:
        attention_hook.remove_hooks()
    
    # 要約プロット作成（データがある場合のみ）
    if all_analysis_results:
        summary_path = os.path.join(output_dir, 'attention_analysis_summary.png')
        create_attention_summary_plot(all_analysis_results, summary_path)
        
        # 位置バイアス単体プロット作成
        bias_path = os.path.join(output_dir, 'positional_bias_analysis.png')
        create_positional_bias_plot(all_analysis_results, bias_path)
    else:
        print("[WARNING] No analysis results available for plotting")
    
    # 結果をJSONで保存
    results_path = os.path.join(output_dir, 'attention_analysis_results.json')
    with open(results_path, 'w') as f:
        # NumPy配列をリストに変換してJSON保存可能にする
        json_results = []
        for result in all_analysis_results:
            json_result = {
                'batch_index': result['batch_index'],
                'point_cloud_blocks': result['point_cloud_blocks'],
                'total_attention': float(result['total_attention']),
                'sequence_length': result['sequence_length'],
                'num_point_blocks': result['num_point_blocks']
            }
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=2)
    
    print(f"[INFO] Attention analysis results saved to: {results_path}")
    
    # 統計サマリー出力
    print("\n" + "="*50)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*50)
    
    if all_analysis_results:
        avg_blocks = np.mean([r['num_point_blocks'] for r in all_analysis_results])
        print(f"Average number of point cloud blocks: {avg_blocks:.1f}")
        
        # 位置別の注意度統計
        position_stats = defaultdict(list)
        for result in all_analysis_results:
            for block in result['point_cloud_blocks']:
                position_stats[block['block_index']].append(block['attention_percentage'])
        
        for pos in sorted(position_stats.keys()):
            avg_attn = np.mean(position_stats[pos])
            std_attn = np.std(position_stats[pos])
            print(f"Position {pos+1}: {avg_attn:.1f}% ± {std_attn:.1f}%")
        
        # 最後の位置への注意集中度
        if len(position_stats) >= 2:
            last_pos = max(position_stats.keys())
            last_attention = np.mean(position_stats[last_pos])
            print(f"\n[KEY FINDING] Last position attention: {last_attention:.1f}%")
            if last_attention > 70:
                print("⚠️  HIGH ATTENTION BIAS DETECTED - Model focuses heavily on last point cloud!")
    
    return all_analysis_results

def main():
    parser = argparse.ArgumentParser(description="Analyze self-attention patterns in PointLLM")
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--lora_dir", type=str, default=None, help="LoRA directory")
    parser.add_argument("--data_path", type=str, default="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat")
    parser.add_argument("--output_dir", type=str, default="./attention_analysis_output", help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to analyze")
    parser.add_argument("--subset_nums", type=int, default=20, help="Subset of dataset to use")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデル読み込み
    print("[INFO] Loading model...")
    model, tokenizer = load_model_with_lora(args.model_name, args.lora_dir)
    
    # データ読み込み
    print("[INFO] Loading dataset...")
    from pointllm.data import ModelNet
    dataset = ModelNet(config_path=None, split="test", subset_nums=args.subset_nums, use_color=True, data_path=args.data_path)
    
    # 簡単なDataLoader（複数オブジェクト用に変換）
    simple_samples = []
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        # 3つのオブジェクトを作成（デモ用）
        point_clouds_list = [sample['point_clouds']]
        # 他の2つはランダム選択
        for j in range(2):
            rand_idx = np.random.randint(len(dataset))
            rand_sample = dataset[rand_idx]
            point_clouds_list.append(rand_sample['point_clouds'])
        
        multi_point_clouds = torch.stack(point_clouds_list, dim=0)
        simple_samples.append({
            'indice': i,
            'point_clouds': multi_point_clouds.unsqueeze(0),  # (1, 3, N, C)
            'labels': [sample['labels'], dataset[rand_idx]['labels'], dataset[(rand_idx+1)%len(dataset)]['labels']]
        })
    
    class SimpleDataLoader:
        def __init__(self, samples):
            self.samples = samples
        def __iter__(self):
            return iter(self.samples)
        def __len__(self):
            return len(self.samples)
    
    dataloader = SimpleDataLoader(simple_samples)
    
    # 注意度分析実行
    print("[INFO] Running attention analysis...")
    results = run_attention_analysis(model, tokenizer, dataloader, args.output_dir, args.num_samples)
    
    print(f"\n[INFO] Analysis complete! Results saved to: {args.output_dir}")
    print(f"[INFO] Check the following files:")
    print(f"  - attention_analysis_summary.png: Overview visualization (both plots)")
    print(f"  - positional_bias_analysis.png: Standalone positional bias analysis")
    print(f"  - attention_heatmap_sample_*.png: Individual heatmaps")
    print(f"  - attention_analysis_results.json: Raw numerical results")

if __name__ == "__main__":
    main() 
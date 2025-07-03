#!/usr/bin/env python3

import torch
from pointllm.model import PointLLMLlamaForCausalLM
from transformers import AutoTokenizer

def demonstrate_lora_safety():
    """LoRAの安全性をデモ"""
    model_name = "RunsenXu/PointLLM_7B_v1.2"
    
    print("=" * 70)
    print("LoRA安全性デモンストレーション")
    print("=" * 70)
    
    # 1. 元のモデルを読み込み
    print("1. 元のPointLLMモデルを読み込み...")
    original_model = PointLLMLlamaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    # 元のモデルの一部パラメータを記録
    layer_name = "model.layers.0.self_attn.q_proj.weight"
    if hasattr(original_model, 'get_parameter'):
        original_weight = original_model.get_parameter(layer_name).clone()
    else:
        # パラメータを手動で取得
        original_weight = None
        for name, param in original_model.named_parameters():
            if name == layer_name:
                original_weight = param.data.clone()
                break
    
    print(f"   元のモデルの {layer_name} の一部:")
    if original_weight is not None:
        print(f"   Shape: {original_weight.shape}")
        print(f"   最初の5x5要素: \n{original_weight[:5, :5]}")
    
    print("\n2. LoRA学習を**シミュレート**（実際の学習は行いません）...")
    
    # PEFTが利用可能な場合のみデモ
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        # LoRA設定
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # LoRAモデル作成
        lora_model = get_peft_model(original_model, lora_config)
        
        print("   ✅ LoRAアダプターを適用しました")
        lora_model.print_trainable_parameters()
        
        # 3. ベースモデルの重みが変更されていないことを確認
        print("\n3. ベースモデルの重みチェック...")
        
        # LoRA適用後でも元の重みにアクセス可能
        base_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight.data
        
        print(f"   LoRA適用後の基底重み {layer_name} の一部:")
        print(f"   Shape: {base_weight.shape}")
        print(f"   最初の5x5要素: \n{base_weight[:5, :5]}")
        
        # 重みの比較
        if torch.equal(original_weight, base_weight):
            print("   ✅ SUCCESS: ベースモデルの重みは全く変更されていません！")
        else:
            print("   ❌ ERROR: 重みが変更されています（これは予期しない動作です）")
        
        # 4. LoRAアダプターの重み確認
        print("\n4. LoRAアダプターの重み:")
        for name, param in lora_model.named_parameters():
            if "lora" in name and param.requires_grad:
                print(f"   {name}: {param.shape} (学習可能)")
                if "lora_A" in name:
                    print(f"      LoRA A行列サンプル: {param.data[:3, :3]}")
                elif "lora_B" in name:
                    print(f"      LoRA B行列サンプル: {param.data[:3, :3]}")
                break
        
        # 5. 推論時の動作説明
        print("\n5. 推論時の動作:")
        print("   出力 = ベース重み × 入力 + LoRA_B × LoRA_A × 入力")
        print("   つまり: W_original × x + (LoRA_B @ LoRA_A) × x")
        print("   ベース重みは読み取り専用、LoRAアダプターのみ学習")
        
    except ImportError:
        print("   PEFTライブラリが利用できません")
        print("   pip install peft でインストールしてください")
    
    print("\n" + "=" * 70)
    print("まとめ:")
    print("• 元のモデルファイルは一切変更されません")
    print("• LoRAアダプター（数MB）のみが新規作成されます") 
    print("• いつでも元のモデルを単独で使用できます")
    print("• 複数のLoRAアダプターを同じベースモデルで使い分け可能")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_lora_safety() 
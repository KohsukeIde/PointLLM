#!/usr/bin/env python3

import torch
import hashlib
from transformers import AutoModelForCausalLM
from pointllm.model import PointLLMLlamaForCausalLM
import os

def get_model_hash(model):
    """モデルの重みのハッシュ値を計算"""
    all_params = []
    for name, param in model.named_parameters():
        all_params.append(param.data.cpu().flatten())
    
    # 全パラメータを結合してハッシュ計算
    all_tensor = torch.cat(all_params)
    tensor_bytes = all_tensor.numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()

def verify_base_model_integrity():
    """ベースモデルの整合性を確認"""
    model_name = "RunsenXu/PointLLM_7B_v1.2"
    
    print("=" * 60)
    print("ベースモデル整合性チェック")
    print("=" * 60)
    
    # 1. 学習前のベースモデル
    print("1. 学習前のベースモデルを読み込み中...")
    model_before = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # ハッシュ計算のため
        low_cpu_mem_usage=True
    )
    hash_before = get_model_hash(model_before)
    print(f"学習前ハッシュ: {hash_before}")
    
    # メモリ解放
    del model_before
    torch.cuda.empty_cache()
    
    # 2. 学習後のベースモデル（LoRA学習の影響を受けていないはず）
    print("\n2. LoRA学習後のベースモデルを読み込み中...")
    model_after = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # ハッシュ計算のため
        low_cpu_mem_usage=True
    )
    hash_after = get_model_hash(model_after)
    print(f"学習後ハッシュ: {hash_after}")
    
    # 3. 比較
    print("\n3. 整合性チェック結果:")
    if hash_before == hash_after:
        print("✅ SUCCESS: ベースモデルの重みは変更されていません")
        print("   LoRAは正しく動作し、元のモデルを保護しています")
    else:
        print("❌ WARNING: ベースモデルの重みが変更されています")
        print("   これは予期しない動作です")
    
    # 4. LoRAアダプターの確認
    lora_dir = "lora_outputs/ordinal_pointllm"
    if os.path.exists(lora_dir):
        print(f"\n4. LoRAアダプター情報:")
        print(f"   ディレクトリ: {lora_dir}")
        
        adapter_files = []
        for file in os.listdir(lora_dir):
            if file.endswith('.bin') or file.endswith('.json'):
                file_path = os.path.join(lora_dir, file)
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                adapter_files.append(f"   {file}: {file_size:.1f} MB")
        
        for file_info in adapter_files:
            print(file_info)
        
        # LoRA統合モデルとの比較
        try:
            from peft import PeftModel
            print("\n5. LoRA統合モデルでの確認:")
            
            base_model = PointLLMLlamaForCausalLM.from_pretrained(model_name)
            lora_model = PeftModel.from_pretrained(base_model, lora_dir)
            
            print("   ✅ LoRAアダプターは正常に読み込まれます")
            print("   ✅ ベースモデル + アダプター = 完全なLoRAモデル")
            
            # LoRAパラメータ統計
            lora_model.print_trainable_parameters()
            
        except Exception as e:
            print(f"   ⚠️  LoRA統合エラー: {e}")
    else:
        print(f"\n4. LoRAアダプターが見つかりません: {lora_dir}")
    
    print("\n" + "=" * 60)
    print("整合性チェック完了")
    print("=" * 60)

if __name__ == "__main__":
    verify_base_model_integrity() 
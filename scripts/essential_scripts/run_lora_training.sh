#!/bin/bash

# H200 GPU向け LoRA Training Script (最高速設定)
# 設定: batch_size=4, grad_accum=2, workers=8
# 予想: 18分, 100GB

set -e  # Exit on any error

PROJECT_ROOT="/groups/gag51404/ide/PointLLM"
echo "=========================================="
echo "PointLLM LoRA Training (最高速設定)"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "設定: batch=4, grad_accum=2, workers=8"
echo "予想時間: 18分"
echo "メモリ使用量: ~100GB / 140GB"
echo "=========================================="

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Error: No GPU detected. Please run on compute nodes."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "$PROJECT_ROOT"

# Check dataset
DATASET_FILE="$PROJECT_ROOT/data/cap3d_ordinal_training/ordinal_dataset_cap3d.json"
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset not found: $DATASET_FILE"
    echo "Please generate dataset first with:"
    echo "python ordinal_reference_project/dataset_generation/ordinal_dataset_generator_cap3d.py"
    exit 1
fi

echo "Dataset found. Stats:"
cat data/cap3d_ordinal_training/dataset_stats.json

# Install dependencies if needed
echo "=========================================="
echo "Checking Dependencies"
echo "=========================================="
python -c "import peft, torch; print('✅ PEFT and PyTorch available')" || {
    echo "Installing required packages..."
    pip install peft==0.10.0 bitsandbytes==0.43.1
}

# Display training configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "• Dataset: 1,980 samples (2K ordinal reference)"
echo "• Epochs: 2 (optimal for LoRA)"
echo "• Batch Size: 4 x 2 grad_accum = 8 effective"
echo "• Learning Rate: 1e-4"
echo "• Target Modules: q_proj, k_proj, v_proj, o_proj"
echo "• Expected Steps: ~496"
echo "• Expected Time: ~18 minutes"
echo "• Memory Usage: ~100GB"

# Start training
echo "=========================================="
echo "Starting LoRA Training"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python ordinal_reference_project/lora_training/train_ordinal_lora.py \
    --config configs/lora_ordinal_config_ultra_fast.json \
    --seed 42

echo "=========================================="
echo "Training Completed!"
echo "=========================================="

# Check output
OUTPUT_DIR="lora_outputs/cap3d_ordinal_2k_ultra_fast"
if [ -d "$OUTPUT_DIR" ]; then
    echo "LoRA Adapter saved to: $OUTPUT_DIR"
    echo "Adapter size:"
    ls -lh "$OUTPUT_DIR"/adapter_model.bin 2>/dev/null || echo "Adapter model not found"
    
    echo "Checkpoints:"
    ls -la "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "No checkpoints found"
else
    echo "Warning: Output directory not found: $OUTPUT_DIR"
fi

echo "=========================================="
echo "Base Model Integrity Check"
echo "=========================================="
python scripts/essential_scripts/verify_base_model_integrity.py

echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Evaluate LoRA model:"
echo "   python pointllm/eval/eval_modelnet_multi.py \\"
echo "     --model_name RunsenXu/PointLLM_7B_v1.2 \\"
echo "     --lora_dir $OUTPUT_DIR \\"
echo "     --output_dir evaluation/lora_trained"
echo ""
echo "2. Compare with baseline:"
echo "   python scripts/essential_scripts/compare_evaluation_results.py \\"
echo "     --base_results evaluation/baseline/ \\"
echo "     --lora_results evaluation/lora_trained/"
echo ""
echo "3. LoRA safety demo:"
echo "   python scripts/essential_scripts/lora_safety_demo.py"
echo "==========================================" 
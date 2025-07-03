#!/bin/bash

# Test Ordinal Training Pipeline with Evaluation
# Run this script on compute nodes: qsub -I -P gag51402 -q rt_HG -l select=1 -l walltime=6:00:00

set -e  # Exit on any error

PROJECT_ROOT="/groups/gag51404/ide/PointLLM"
SUBSET_RATIO=0.1  # 10% for testing
EVAL_SUBSET_NUMS=100  # 100 samples for evaluation

echo "=========================================="
echo "PointLLM Ordinal Training Test Pipeline"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Training Subset Ratio: $SUBSET_RATIO"
echo "Evaluation Subset: $EVAL_SUBSET_NUMS samples"
echo "=========================================="

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Error: No GPU detected. Please run on compute nodes."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd "$PROJECT_ROOT"

# Install dependencies
echo "=========================================="
echo "Step 1: Installing Dependencies"
echo "=========================================="
python -c "import peft" 2>/dev/null || {
    echo "Installing PEFT library..."
    pip install peft==0.10.0 bitsandbytes==0.43.1
}

# Generate ordinal training dataset
echo "=========================================="
echo "Step 2: Generating Ordinal Training Dataset"
echo "=========================================="
OUTPUT_DIR="$PROJECT_ROOT/data/ordinal_training_data"

if [ ! -f "$OUTPUT_DIR/ordinal_dataset.jsonl" ]; then
    python pointllm/data/ordinal_dataset_generator.py \
        --output_dir "$OUTPUT_DIR" \
        --subset_ratio "$SUBSET_RATIO" \
        --use_color \
        --seed 42
else
    echo "Ordinal dataset already exists."
fi

# Analyze dataset
python pointllm/data/ordinal_data_storage.py \
    --dataset_dir "$OUTPUT_DIR" \
    --analyze

echo "=========================================="
echo "Step 3: Baseline Evaluation (Before LoRA)"
echo "=========================================="

# Evaluate base model with ordinal prompts
echo "3.1 Evaluating base model with eval_modelnet_multi..."
python pointllm/eval/eval_modelnet_multi.py \
    --model_name "RunsenXu/PointLLM_7B_v1.2" \
    --output_dir "evaluation/baseline" \
    --subset_nums "$EVAL_SUBSET_NUMS" \
    --prompt_index 2 \
    --num_objects 3

# Evaluate with CLIP
echo "3.2 Evaluating base model with CLIP evaluation..."
python experiments/scripts/2_classification_check_CLIP.py \
    --output_dir "evaluation/baseline_clip" \
    --subset_nums "$EVAL_SUBSET_NUMS"

echo "=========================================="
echo "Step 4: LoRA Training"
echo "=========================================="

LORA_OUTPUT_DIR="$PROJECT_ROOT/lora_outputs/ordinal_pointllm"
CONFIG_FILE="$PROJECT_ROOT/configs/lora_ordinal_config.json"

# Update config paths
python -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['data_args']['dataset_dir'] = '$OUTPUT_DIR'
config['training_args']['output_dir'] = '$LORA_OUTPUT_DIR'
config['training_args']['num_train_epochs'] = 2  # Reduce for testing
config['training_args']['save_steps'] = 50       # Save more frequently
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print('LoRA configuration updated for testing')
"

mkdir -p "$LORA_OUTPUT_DIR"

echo "Starting LoRA training..."
python pointllm/train/train_ordinal_lora.py \
    --config "$CONFIG_FILE" \
    --seed 42

echo "LoRA training completed!"

echo "=========================================="
echo "Step 5: Post-Training Evaluation"
echo "=========================================="

# Evaluate LoRA model with ordinal dataset
echo "5.1 Evaluating LoRA model with ordinal dataset..."
python pointllm/eval/eval_modelnet_multi_lora.py \
    --dataset_dir "$OUTPUT_DIR" \
    --lora_dir "$LORA_OUTPUT_DIR" \
    --eval_mode dataset \
    --output_dir "evaluation/lora_ordinal" \
    --subset_ratio 0.5

# Evaluate LoRA model with custom prompts
echo "5.2 Evaluating LoRA model with custom prompts..."
python pointllm/eval/eval_modelnet_multi_lora.py \
    --dataset_dir "$OUTPUT_DIR" \
    --lora_dir "$LORA_OUTPUT_DIR" \
    --eval_mode custom \
    --output_dir "evaluation/lora_custom" \
    --subset_ratio 0.5

# Evaluate with eval_modelnet_multi (if LoRA integration is added)
echo "5.3 Evaluating LoRA model with eval_modelnet_multi..."
# Note: This might require modification to eval_modelnet_multi.py to support LoRA

echo "=========================================="
echo "Step 6: Base Model Integrity Check"
echo "=========================================="

python scripts/verify_base_model_integrity.py

echo "=========================================="
echo "Step 7: Results Summary"
echo "=========================================="

echo "Training completed! Results summary:"
echo "1. Baseline evaluation: evaluation/baseline/"
echo "2. Baseline CLIP evaluation: evaluation/baseline_clip/"
echo "3. LoRA model: $LORA_OUTPUT_DIR"
echo "4. LoRA ordinal evaluation: evaluation/lora_ordinal/"
echo "5. LoRA custom evaluation: evaluation/lora_custom/"

echo "Checking file sizes:"
echo "Base model cache:"
ls -lh ~/.cache/huggingface/hub/models--RunsenXu--PointLLM_7B_v1.2/snapshots/*/pytorch_model*.bin 2>/dev/null || echo "Base model not in cache"

echo "LoRA adapter:"
ls -lh "$LORA_OUTPUT_DIR"/adapter_model.bin 2>/dev/null || echo "LoRA adapter not found"

echo "=========================================="
echo "Analysis Commands:"
echo "=========================================="
echo "Compare results with:"
echo "1. Check ordinal accuracy improvement:"
echo "   cat evaluation/lora_ordinal/ordinal_evaluation_modedataset.json | jq '.summary'"
echo ""
echo "2. Analyze base vs LoRA performance:"
echo "   python scripts/compare_evaluation_results.py"
echo ""
echo "3. View LoRA training statistics:"
echo "   ls -la $LORA_OUTPUT_DIR"
echo "=========================================="

echo "Test pipeline completed successfully!" 
#!/bin/bash

# Cap3D-based Ordinal Training Pipeline with Evaluation
# Run this script on compute nodes: qsub -I -P gag51402 -q rt_HG -l select=1 -l walltime=6:00:00

set -e  # Exit on any error

PROJECT_ROOT="/groups/gag51404/ide/PointLLM"
SUBSET_RATIO=0.01  # 1% for quick testing
EVAL_SUBSET_NUMS=50  # 50 samples for evaluation

echo "=========================================="
echo "PointLLM Cap3D Ordinal Training Pipeline"
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

# Generate Cap3D ordinal training dataset
echo "=========================================="
echo "Step 2: Generating Cap3D Ordinal Dataset"
echo "=========================================="
OUTPUT_DIR="$PROJECT_ROOT/data/cap3d_ordinal_training"

if [ ! -f "$OUTPUT_DIR/ordinal_dataset_cap3d.json" ]; then
    python pointllm/data/ordinal_dataset_generator_cap3d.py \
        --data_dir "data/anno_data" \
        --output_dir "$OUTPUT_DIR" \
        --use_complex \
        --subset_ratio "$SUBSET_RATIO" \
        --seed 42
else
    echo "Cap3D ordinal dataset already exists."
fi

# Check dataset stats
echo "Dataset Statistics:"
if [ -f "$OUTPUT_DIR/dataset_stats.json" ]; then
    cat "$OUTPUT_DIR/dataset_stats.json"
fi

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

echo "=========================================="
echo "Step 4: LoRA Training Configuration"
echo "=========================================="

LORA_OUTPUT_DIR="$PROJECT_ROOT/lora_outputs/cap3d_ordinal_pointllm"
CONFIG_FILE="$PROJECT_ROOT/configs/lora_ordinal_config.json"

# Update config paths
python -c "
import json
config_file = '$CONFIG_FILE'
with open(config_file, 'r') as f:
    config = json.load(f)
config['data_args']['dataset_dir'] = '$OUTPUT_DIR'
config['data_args']['dataset_file'] = 'ordinal_dataset_cap3d.json'
config['training_args']['output_dir'] = '$LORA_OUTPUT_DIR'
config['training_args']['num_train_epochs'] = 1  # Quick test
config['training_args']['save_steps'] = 50
config['training_args']['per_device_train_batch_size'] = 1  # Reduce memory usage
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
print('LoRA configuration updated for Cap3D training')
"

mkdir -p "$LORA_OUTPUT_DIR"

echo "=========================================="
echo "Step 5: LoRA Training"
echo "=========================================="

echo "Starting LoRA training with Cap3D data..."
python pointllm/train/train_ordinal_lora.py \
    --config "$CONFIG_FILE" \
    --seed 42

echo "LoRA training completed!"

echo "=========================================="
echo "Step 6: Post-Training Evaluation"
echo "=========================================="

# Evaluate LoRA model with eval_modelnet_multi
echo "6.1 Evaluating LoRA model with eval_modelnet_multi..."
python pointllm/eval/eval_modelnet_multi.py \
    --model_name "RunsenXu/PointLLM_7B_v1.2" \
    --lora_dir "$LORA_OUTPUT_DIR" \
    --output_dir "evaluation/lora_enhanced" \
    --subset_nums "$EVAL_SUBSET_NUMS" \
    --prompt_index 2 \
    --num_objects 3

echo "=========================================="
echo "Step 7: Base Model Integrity Check"
echo "=========================================="

python scripts/verify_base_model_integrity.py

echo "=========================================="
echo "Step 8: Results Summary"
echo "=========================================="

echo "Training completed! Results summary:"
echo "1. Cap3D dataset: $OUTPUT_DIR"
echo "2. Baseline evaluation: evaluation/baseline/"
echo "3. LoRA model: $LORA_OUTPUT_DIR"
echo "4. LoRA evaluation: evaluation/lora_enhanced/"

echo "Checking file sizes:"
echo "LoRA adapter:"
ls -lh "$LORA_OUTPUT_DIR"/adapter_model.bin 2>/dev/null || echo "LoRA adapter not found"

echo "=========================================="
echo "Analysis Commands:"
echo "=========================================="
echo "Compare results with:"
echo "1. python scripts/compare_evaluation_results.py \\"
echo "     --base_results evaluation/baseline/ModelNet_multi_classification_prompt2_obj3.json \\"
echo "     --lora_results evaluation/lora_enhanced/ModelNet_multi_classification_prompt2_obj3.json \\"
echo "     --output_dir evaluation/comparison"
echo ""
echo "2. View Cap3D dataset:"
echo "   head -3 $OUTPUT_DIR/ordinal_dataset_cap3d.json"
echo ""
echo "3. Check LoRA training logs:"
echo "   ls -la $LORA_OUTPUT_DIR"
echo "=========================================="

echo "Cap3D ordinal training pipeline completed successfully!" 
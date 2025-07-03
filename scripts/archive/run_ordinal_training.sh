#!/bin/bash

# Complete Ordinal Training Pipeline for PointLLM
# This script should be run on compute nodes (qsub -I -P gag51402 -q rt_HG -l select=1 -l walltime=6:00:00)

set -e  # Exit on any error

# Default parameters
SUBSET_RATIO=${1:-0.1}  # Default to 10% for testing, use 1.0 for full dataset
PROJECT_ROOT="/groups/gag51404/ide/PointLLM"
OUTPUT_DIR="$PROJECT_ROOT/data/ordinal_training_data"
LORA_OUTPUT_DIR="$PROJECT_ROOT/lora_outputs/ordinal_pointllm"
CONFIG_FILE="$PROJECT_ROOT/configs/lora_ordinal_config.json"

echo "=========================================="
echo "PointLLM Ordinal Training Pipeline"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Dataset Directory: $OUTPUT_DIR"
echo "LoRA Output Directory: $LORA_OUTPUT_DIR"
echo "Subset Ratio: $SUBSET_RATIO"
echo "=========================================="

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Error: No GPU detected or nvidia-smi not available"
    echo "This script requires GPU access. Please run on compute nodes."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Change to project directory
cd "$PROJECT_ROOT"

# Step 1: Install required dependencies
echo "=========================================="
echo "Step 1: Installing required dependencies"
echo "=========================================="

# Check if peft is installed
python -c "import peft" 2>/dev/null || {
    echo "Installing PEFT library..."
    pip install peft==0.10.0 bitsandbytes==0.43.1
}

# Step 2: Generate ordinal dataset
echo "=========================================="
echo "Step 2: Generating ordinal dataset"
echo "=========================================="

if [ ! -f "$OUTPUT_DIR/ordinal_dataset.jsonl" ]; then
    echo "Generating new ordinal dataset with subset_ratio=$SUBSET_RATIO..."
    python pointllm/data/ordinal_dataset_generator.py \
        --output_dir "$OUTPUT_DIR" \
        --subset_ratio "$SUBSET_RATIO" \
        --use_color \
        --seed 42
else
    echo "Ordinal dataset already exists. Skipping generation."
fi

# Step 3: Analyze generated dataset
echo "=========================================="
echo "Step 3: Analyzing generated dataset"
echo "=========================================="

python pointllm/data/ordinal_data_storage.py \
    --dataset_dir "$OUTPUT_DIR" \
    --analyze

# Step 4: Update LoRA config with correct paths
echo "=========================================="
echo "Step 4: Updating LoRA configuration"
echo "=========================================="

# Update dataset path in config
python -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['data_args']['dataset_dir'] = '$OUTPUT_DIR'
config['training_args']['output_dir'] = '$LORA_OUTPUT_DIR'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print('LoRA configuration updated')
"

# Step 5: Run LoRA training
echo "=========================================="
echo "Step 5: Starting LoRA training"
echo "=========================================="

# Create output directory
mkdir -p "$LORA_OUTPUT_DIR"

# Monitor GPU memory
echo "GPU memory before training:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Run training
python pointllm/train/train_ordinal_lora.py \
    --config "$CONFIG_FILE" \
    --seed 42

echo "=========================================="
echo "Step 6: Training completed"
echo "=========================================="

echo "GPU memory after training:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo "Generated files:"
echo "Dataset: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/"

echo "LoRA model: $LORA_OUTPUT_DIR"
ls -la "$LORA_OUTPUT_DIR/"

echo "=========================================="
echo "Step 7: Quick evaluation test"
echo "=========================================="

# Test the trained model with a simple prompt
python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from peft import PeftModel
from pointllm.model import PointLLMLlamaForCausalLM
from transformers import AutoTokenizer

print('Loading trained LoRA model for testing...')
base_model = PointLLMLlamaForCausalLM.from_pretrained('RunsenXu/PointLLM_7B_v1.2')
model = PeftModel.from_pretrained(base_model, '$LORA_OUTPUT_DIR')
tokenizer = AutoTokenizer.from_pretrained('RunsenXu/PointLLM_7B_v1.2')

print('LoRA model loaded successfully!')
print('Training pipeline completed!')
"

echo "=========================================="
echo "Ordinal training pipeline completed!"
echo "=========================================="
echo "Next steps:"
echo "1. Test the model with: pointllm/eval/eval_modelnet_multi_lora.py"
echo "2. Run full evaluation on test set"
echo "3. Compare performance with base model"
echo "==========================================" 
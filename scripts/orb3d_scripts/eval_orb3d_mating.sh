#!/bin/bash

# ORB3D Shape Mating Evaluation Script

MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"
DATA_PATH="/groups/gag51402/datasets/ORB3D/objaverse_PWN_filtered/objaverse"
OUTPUT_DIR="./evaluation_orb3d"

echo "Starting ORB3D Shape Mating Evaluation..."

# Binary mating test (all prompts)
echo "Running Binary Mating Tests..."
for prompt_idx in {0..4}; do
    echo "  Prompt $prompt_idx..."
    python pointllm/eval/eval_orb3d_mating.py \
        --data_path $DATA_PATH \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --test_type binary \
        --prompt_index $prompt_idx \
        --npoints 8192
done

# Multi-choice mating test (all prompts)
echo "Running Multi-choice Mating Tests..."
for prompt_idx in {0..4}; do
    echo "  Prompt $prompt_idx..."
    python pointllm/eval/eval_orb3d_mating.py \
        --data_path $DATA_PATH \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --test_type multi_choice \
        --prompt_index $prompt_idx \
        --npoints 8192
done

echo "All evaluations completed. Results saved in $OUTPUT_DIR"

# Generate summary report
echo "Generating summary report..."
python scripts/summarize_orb3d_results.py --results_dir $OUTPUT_DIR 
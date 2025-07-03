#!/bin/bash

# Ordinal Dataset Generation Script for PointLLM
# Usage: ./generate_ordinal_dataset.sh [subset_ratio]

set -e  # Exit on any error

# Set default parameters
SUBSET_RATIO=${1:-0.1}  # Default to 10% for testing
PROJECT_ROOT="/groups/gag51404/ide/PointLLM"
OUTPUT_DIR="$PROJECT_ROOT/data/ordinal_training_data"
DATA_PATH="$PROJECT_ROOT/data/modelnet40_data/modelnet40_train_8192pts_fps.dat"

echo "=========================================="
echo "PointLLM Ordinal Dataset Generation"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Path: $DATA_PATH"
echo "Subset Ratio: $SUBSET_RATIO"
echo "=========================================="

# Check if we're in the correct directory
cd "$PROJECT_ROOT"

# Check if ModelNet data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: ModelNet data file not found at $DATA_PATH"
    echo "Please check the data path and try again."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the dataset generation
echo "Starting dataset generation..."
python pointllm/data/ordinal_dataset_generator.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --subset_ratio "$SUBSET_RATIO" \
    --use_color \
    --seed 42

echo "=========================================="
echo "Dataset generation completed!"
echo "Output directory: $OUTPUT_DIR"

# Show generated files
echo "Generated files:"
ls -la "$OUTPUT_DIR/"
if [ -d "$OUTPUT_DIR/point_clouds" ]; then
    echo "Point cloud files: $(ls -1 $OUTPUT_DIR/point_clouds/ | wc -l) files"
fi

# Run analysis
echo "=========================================="
echo "Running dataset analysis..."
python pointllm/data/ordinal_data_storage.py \
    --dataset_dir "$OUTPUT_DIR" \
    --analyze

echo "=========================================="
echo "Script completed successfully!" 
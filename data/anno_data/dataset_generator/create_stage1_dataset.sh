#!/bin/bash

# Stage1 Dataset Generation Script for PointLLM
# 複数点群入力対応のbrief descriptionデータセット生成

set -e

echo "======================================"
echo "Stage1 Dataset Generation for PointLLM"
echo "======================================"

# 基本設定
PROJECT_ROOT="/groups/gag51404/ide/PointLLM"
DATA_DIR="${PROJECT_ROOT}/data/anno_data"
OUTPUT_DIR="${PROJECT_ROOT}/data/anno_data"
SCRIPT_PATH="${PROJECT_ROOT}/data/anno_data/dataset_generator/create_stage1_dataset.py"

# 生成パラメータ
NUM_SAMPLES=660000
SEED=42
DISTRIBUTION="single:220000,two:220000,three:220000"
DIVERSITY_METHOD="random"
USE_POSITION_TOKENS="--use_position_tokens"

echo "Project Root: ${PROJECT_ROOT}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of Samples: ${NUM_SAMPLES}"
echo "Distribution: ${DISTRIBUTION}"
echo "Diversity Method: ${DIVERSITY_METHOD}"
echo ""

# データディレクトリの確認
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# 入力ファイルの確認
INPUT_FILE="${DATA_DIR}/PointLLM_brief_description_660K_filtered.json"
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: Input file not found: ${INPUT_FILE}"
    echo "Please ensure the brief description dataset is available."
    exit 1
fi

# 出力ディレクトリの作成
mkdir -p "${OUTPUT_DIR}"

# Python環境の確認
echo "Checking Python environment..."
python3 --version
echo ""

# データセット生成の実行
echo "Starting Stage1 dataset generation..."
echo "This may take several minutes for ${NUM_SAMPLES} samples..."
echo ""

cd "${PROJECT_ROOT}"

python3 "${SCRIPT_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --total_samples "${NUM_SAMPLES}" \
    --distribution "${DISTRIBUTION}" \
    --diversity_method "${DIVERSITY_METHOD}" \
    ${USE_POSITION_TOKENS} \
    --seed "${SEED}"

echo ""
echo "======================================"
echo "Stage1 Dataset Generation Completed!"
echo "======================================"
echo ""
echo "Generated files:"
echo "- Dataset: ${OUTPUT_DIR}/brief_description_stage1_multi_pc.json"
echo "- Statistics: ${OUTPUT_DIR}/brief_description_stage1_multi_pc_stats.json"
echo ""
echo "Next steps:"
echo "1. Verify the generated dataset structure"
echo "2. Run Stage1 training with this dataset"
echo "3. Use trained Stage1 model for Stage2 training"
echo ""

# ファイルサイズと行数の表示
OUTPUT_FILE="${OUTPUT_DIR}/brief_description_stage1_multi_pc.json"
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Dataset file size: $(du -h "${OUTPUT_FILE}" | cut -f1)"
    echo "Dataset samples: $(grep -c '"object_id"' "${OUTPUT_FILE}" 2>/dev/null || echo 'Could not count samples')"
fi 
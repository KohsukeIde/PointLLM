#!/bin/bash

# Stage2 Dataset Generation Script for PointLLM
# 複数点群入力対応のデータセット生成

set -e

echo "======================================"
echo "Stage2 Dataset Generation for PointLLM"
echo "======================================"

# 基本設定
PROJECT_ROOT="/groups/gag51404/ide/PointLLM"
DATA_DIR="${PROJECT_ROOT}/data/anno_data"
OUTPUT_DIR="${PROJECT_ROOT}/data/anno_data"
SCRIPT_PATH="${PROJECT_ROOT}data/anno_data/dataset_generator/create_stage2_dataset.py"

# 生成パラメータ
NUM_SAMPLES=70000
SEED=42

echo "Project Root: ${PROJECT_ROOT}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of Samples: ${NUM_SAMPLES}"
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
echo "Starting Stage2 dataset generation..."
echo "This may take several minutes..."
echo ""

cd "${PROJECT_ROOT}"

python3 "${SCRIPT_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}"

echo ""
echo "======================================"
echo "Stage2 Dataset Generation Completed!"
echo "======================================"
echo ""
echo "Generated files:"
echo "- Dataset: ${OUTPUT_DIR}/PointLLM_complex_instruction_stage2_multi_pc_70K.json"
echo "- Statistics: ${OUTPUT_DIR}/PointLLM_complex_instruction_stage2_multi_pc_70K_stats.json"
echo ""
echo "Next steps:"
echo "1. Verify the generated dataset structure"
echo "2. Modify the data loader for multi-object support"
echo "3. Create Stage2 training script"
echo ""

# ファイルサイズと行数の表示
OUTPUT_FILE="${OUTPUT_DIR}/PointLLM_complex_instruction_stage2_multi_pc_70K.json"
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Dataset file size: $(du -h "${OUTPUT_FILE}" | cut -f1)"
    echo "Dataset samples: $(grep -c '"object_ids"' "${OUTPUT_FILE}" 2>/dev/null || echo 'Could not count samples')"
fi 
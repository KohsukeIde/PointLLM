#!/bin/bash

# プロジェクトルートへのパスを定義（絶対パスを使用）
PROJECT_ROOT="/groups/gag51404/ide/PointLLM"

# ベースディレクトリの設定
NOISY_DIR="${PROJECT_ROOT}/data/modelnet40_data/ModelNet40_noisy"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"
MODEL_BASENAME=$(basename "$MODEL_NAME")
PROMPT_INDEX=0

# 結果保存用のログディレクトリを作成
LOG_DIR="${PROJECT_ROOT}/evaluation_logs"
mkdir -p $LOG_DIR

echo "Starting evaluation of all noisy variants..."
echo "Results will be saved in ${PROJECT_ROOT}/evaluation/${MODEL_BASENAME}"

# カレントディレクトリをプロジェクトルートに変更
cd "$PROJECT_ROOT"

# 各バリアントディレクトリを処理
for variant_dir in "$NOISY_DIR"/*; do
    if [ -d "$variant_dir" ]; then  # ディレクトリであるか確認
        variant_name=$(basename "$variant_dir")
        dat_file="$variant_dir/modelnet40_test_8192pts_fps_noisy.dat"
        
        # 出力JSONファイルのパスを予測（noise_oodサブディレクトリを含む）
        output_json="${PROJECT_ROOT}/evaluation/${MODEL_BASENAME}/noise_ood/ModelNet_classification_${variant_name}_prompt${PROMPT_INDEX}.json"
        
        # noise_oodディレクトリが存在しない場合は作成
        mkdir -p "${PROJECT_ROOT}/evaluation/${MODEL_BASENAME}/noise_ood/"

        if [ -f "$output_json" ]; then
            echo "Skipping $variant_name - result file already exists: $output_json"
            continue
        fi
        
        if [ -f "$dat_file" ]; then  # ファイルが存在するか確認
            echo "-----------------------------------------------"
            echo "Processing variant: $variant_name"
            echo "Data file: $dat_file"
            echo "-----------------------------------------------"
            
            # 実行時間計測のために開始時刻を記録
            start_time=$(date +%s)
            
            # 評価の実行（絶対パスを使用）
            python "${PROJECT_ROOT}/pointllm/eval/eval_modelnet_cls.py" \
                --model_name "$MODEL_NAME" \
                --output_dir "${PROJECT_ROOT}/evaluation/${MODEL_BASENAME}/noise_ood/" \
                --data_path "$dat_file" \
                --prompt_index $PROMPT_INDEX \
                2>&1 | tee "${LOG_DIR}/${variant_name}.log"
            
            # 終了時刻を記録し、実行時間を計算
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            echo "Completed evaluation of $variant_name in $duration seconds"
            echo ""
        else
            echo "Warning: No .dat file found in $variant_dir"
        fi
    fi
done

echo "All evaluations completed!"
echo "See ${PROJECT_ROOT}/evaluation/${MODEL_BASENAME} for results."

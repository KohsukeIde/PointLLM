#!/bin/bash
# run_missing_judges.sh
# 未処理のPromptX (0-6)に対してood_judge_internvlを実行

BASE_DIR="/groups/gag51404/ide/PointLLM"
RESULTS_DIR="${BASE_DIR}/judge_results"
OOD_DIR="${BASE_DIR}/evaluation/prompt_ood/PointLLM_7B_v1.2"  # 修正した入力パス

# 結果ディレクトリが存在しなければ作成
mkdir -p ${RESULTS_DIR}

# 0-6の各プロンプトをチェック
for X in {0..6}; do
  # 判定結果ファイルのパス
  JUDGED_JSON="${RESULTS_DIR}/ModelNet_OOD_prompt${X}_judged.json"
  JUDGED_CSV="${RESULTS_DIR}/ModelNet_OOD_prompt${X}_judged.csv"
  
  # 入力となるOOD JSONファイルパス
  OOD_JSON="${OOD_DIR}/ModelNet_OOD_prompt${X}.json"
  
  # JSONとCSVの両方が存在しない場合にのみ実行
  if [ ! -f "$JUDGED_JSON" ] || [ ! -f "$JUDGED_CSV" ]; then
    echo "Processing prompt ${X}..."
    
    # 入力ファイルの存在確認
    if [ ! -f "$OOD_JSON" ]; then
      echo "ERROR: Input file not found: ${OOD_JSON}"
      continue
    fi
    
    # ood_judge_internvl.pyを実行
    python -u "${BASE_DIR}/pointllm/eval/ood_judge_internvl.py" \
      --ood_json "${OOD_JSON}" \
      --out_dir "${RESULTS_DIR}" \
      --render_dir "${RESULTS_DIR}/render_prompt${X}"
    
    echo "Completed prompt ${X}"
  else
    echo "Skipping prompt ${X} (already processed)"
  fi
done

echo "All missing prompts processed."
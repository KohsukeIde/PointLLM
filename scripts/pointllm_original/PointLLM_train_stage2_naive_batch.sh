#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gag51402
#PBS -N PointLLM_train_stage2_naive
#PBS -o pointllm_train_stage2_naive.out
#PBS -e pointllm_train_stage2_naive.err

# 環境設定
source /etc/profile.d/modules.sh
module load cuda/12.6

# 作業ディレクトリに移動
cd ../../groups/gag51404/ide/

# Python仮想環境をアクティベート
source venv/pointllm/bin/activate

# PointLLMディレクトリに移動
cd PointLLM/

# ランダムなマスターポートを生成
master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))

# ファイル名取得（バッチジョブ用に固定）
filename="PointLLM_train_stage2_naive_batch"

dir_path=PointLLM

# Stage 2 training for multi-point cloud understanding
model_name_or_path=outputs/PointLLM_train_stage1/PointLLM_train_stage1_original
data_path=data/objaverse_data
anno_path=data/anno_data/complex_instruction_stage2_multi_pc_70K_gpt.json
output_dir=outputs/PointLLM_train_stage2/${filename}

# トレーニング実行
PYTHONPATH=$dir_path:$PYTHONPATH \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "no" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm False \
    --fix_pointnet True \
    --report_to wandb \
    --run_name $filename \
    --gradient_checkpointing True \
    --stage_2 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --conversation_types "simple_description" \
    --use_color True 
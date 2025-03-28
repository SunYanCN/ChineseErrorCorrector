#!/bin/bash

# 设置默认参数
TRAIN_FILE="../data/business_data/train.json"
DEV_FILE="../data/business_data/dev.json"
MODEL_NAME="../pre_model/ChineseErrorCorrector2-7B"
OUTPUT_DIR="../data/business_data/model_output"
CACHE_DIR="../data/business_data/cache_dir"

# 运行训练脚本
python run.py \
    --train_file $TRAIN_FILE \
    --dev_file $DEV_FILE \
    --model_type auto \
    --model_name $MODEL_NAME \
    --do_train True \
    --do_predict True \
    --output_dir $OUTPUT_DIR \
    --device_map auto \
    --bf16 True \
    --device cuda \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --logging_steps 50 \
    --max_steps -1 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing True \
    --torch_compile False \
    --gradient_accumulation_steps 1 \
    --warmup_steps 50 \
    --save_steps 1000 \
    --optimizer adamw_torch \
    --save_strategy steps \
    --eval_steps 1000 \
    --save_total_limit 10 \
    --remove_unused_columns False \
    --report_to tensorboard \
    --overwrite_output_dir True \
    --max_eval_samples 1000 \
    --peft_type LORA \
    --use_peft True \
    --lora_target_modules all \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --no_cache False \
    --dataset_class None \
    --cache_dir $CACHE_DIR \
    --preprocessing_num_workers 4 \
    --reprocess_input_data True \
    --resume_from_checkpoint $OUTPUT_DIR \
    --prompt_template_name qwen \
    --max_seq_length 512 \
    --max_length 512

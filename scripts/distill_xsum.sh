export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=3
accelerate launch --config_file=config.yaml distill.py \
    --model_name_or_path "t5-base" \
    --teacher_path "google/flan-t5-xl" \
    --train_file "data/xsum/train.csv" \
    --validation_file "data/xsum/test.csv" \
    --test_file "data/xsum/test.csv" \
    --max_seq_length 512 \
    --max_answer_length 512 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_beams 1 \
    --step_per_update 12 \
    --doc_stride 128 \
    --seed 0 \
    --output_dir "results/xsum/seqkd" \
    --source_prefix "summarize: " \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 32 \
    --train_type "distill_seqkd" \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --report_to "wandb"

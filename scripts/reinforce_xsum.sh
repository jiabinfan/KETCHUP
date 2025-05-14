export TOKENIZERS_PARALLELISM=false
N_step=2
DATASET=xsum
METHOD=None

OUTPUT_DIR=results/${DATASET}/reinforce_nstep_sampling5_${N_step}

mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR/

CUDA_VISIBLE_DEVICES=3 python3 reinforce.py \
    --model_name_or_path "data/${DATASET}/kl" \
    --teacher_path "google/flan-t5-xl" \
    --train_file "data/${DATASET}/train.csv" \
    --validation_file "data/${DATASET}/valid.csv" \
    --test_file "data/${DATASET}/test.csv" \
    --max_seq_length 512 \
    --max_answer_length 512 \
    --per_device_train_batch_size 1 \
    --num_beams 1 \
    --per_device_eval_batch_size 512 \
    --step_per_update 8 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --source_prefix "summarize: " \
    --learning_rate 1e-5 \
    --num_train_epochs 200 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --top_k 1 \
    --do_sample \
    --n_step ${N_step} \
    --overwrite_output_dir \
    --method $METHOD \
    --report_to "wandb" 2>&1 | tee $OUTPUT_DIR/${METHOD}.log
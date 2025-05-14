#!/bin/bash
export TOKENIZERS_PARALLELISM=false

TOKENIZER=/home/jiabin/flan-t5-xl
reward_model=/home/jiabin/flan-t5-xl
TEMPLATE=templates/summarization_T0.txt
DATA=data/xsum
N_STEP=1


DIR=results/m4_xsum_full/0reinforce_analysis_gq24_n_1
# GPU 0: 0 to 750

# for i in {425..6000..25}
for i in {473..593..8}
do
    echo "i: $i"
    CUDA_VISIBLE_DEVICES=0 python3 cal_return.py \
    --reward-model $reward_model \
    --template $TEMPLATE \
    --tokenizer $TOKENIZER \
    --src $DATA/test.src \
    --gen $DIR/steps_$(printf $i).gen 
done
for i in {1001..1401..8}
do
    echo "i: $i"
    CUDA_VISIBLE_DEVICES=0 python3 cal_return.py \
    --reward-model $reward_model \
    --template $TEMPLATE \
    --tokenizer $TOKENIZER \
    --src $DATA/test.src \
    --gen $DIR/steps_$(printf $i).gen 
done

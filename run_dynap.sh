#!/bin/bash

data_root="./data" # your data path
dataset='pets'  # I/A/R/V/S
logdate='26022215'  # name of the log file
num_p=10  # number of online prompts
selection_p=0.1
lr=0.005
ntx=4
seed=6
arch='ViT-B/16'  #e.g., 'RN50' or 'ViT-B/16'

sh ./scripts/test_dynap.sh ${data_root} ${dataset} ${logdate} ${num_p} ${selection_p} ${lr} ${ntx} ${seed} ${arch}

# python ./dynap_classification.py ./data --test_sets pets \
# -a ViT-B/16 -b 4 --gpu 0 \
# --lr 5e-3 \
# --seed 0 \
# --n_ctx 4 \
# --tpt --ctx_init a_photo_of_a --log_date 20260314 \
# --num_prompts 5 \
# --onlinetpt \
# --selection_p 0.1 \
# --proactive_routing \
# --routing_threshold 0.7 \
# --sample_size 20 \
# > logs/test_dynap_pets_20260314.log 2>&1
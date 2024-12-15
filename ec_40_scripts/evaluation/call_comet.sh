#!/bin/bash

EXPERIMENT_NAME=$1
EXPERIMENT_ID=$2
pt=$3
ROOT_PATH=$4

WORK_PATH=${ROOT_PATH}/ec_40_scripts
SAVE_PATH=${WORK_PATH}/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}

num_gpus=$5
pids=()
declare -A lang_pairs

for tgt in en bg so ca da be bs es uk am hi ro no de cs pt nl mr is ne ur ha sv gu ar fr ru it pl sr sd he af kn bn; do
    lang_pairs["$tgt"]=1
done

for gpu_id in $(seq 0 $((num_gpus - 1))); do
    if [[ ${#lang_pairs[@]} -gt 0 ]]; then
        IFS=',' read -r tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
        unset lang_pairs["$tgt"]
        CUDA_VISIBLE_DEVICES=$gpu_id python evaluation/measure_comet.py $SAVE_PATH $pt $tgt &
        pids[$gpu_id]=$!
    fi
done

while :; do
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if ! kill -0 ${pids[$gpu_id]} 2> /dev/null && [[ ${#lang_pairs[@]} -gt 0 ]]; then
            IFS=',' read -r tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
            unset lang_pairs["$tgt"]
            CUDA_VISIBLE_DEVICES=$gpu_id python evaluation/measure_comet.py $SAVE_PATH $pt $tgt &
            pids[$gpu_id]=$!
        fi
    done
    if [[ ${#lang_pairs[@]} -eq 0 ]]; then
        break 
    fi
    sleep 10 
done

wait


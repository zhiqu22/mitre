#!/bin/bash

EXPERIMENT_NAME=$1
EXPERIMENT_ID=$2
pt=$3
ROOT_PATH=$4
num_gpus=$5

WORK_PATH=${ROOT_PATH}/mitre
SAVE_PATH=${WORK_PATH}/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}

pids=()
declare -A lang_pairs

IFS=" " read -r -a languages <<< $(echo $(python scripts/functions.py get_languages) | tr -d '[],' | tr -d "'")
for tgt in "${languages[@]}"; do
    lang_pairs["$tgt"]=1
done

for gpu_id in $(seq 0 $((num_gpus - 1))); do
    if [[ ${#lang_pairs[@]} -gt 0 ]]; then
        IFS=',' read -r tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
        unset lang_pairs["$tgt"]
        CUDA_VISIBLE_DEVICES=$gpu_id python scripts/measure_comet.py $SAVE_PATH $pt $tgt &
        pids[$gpu_id]=$!
    fi
done

while :; do
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if ! kill -0 ${pids[$gpu_id]} 2> /dev/null && [[ ${#lang_pairs[@]} -gt 0 ]]; then
            IFS=',' read -r tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
            unset lang_pairs["$tgt"]
            CUDA_VISIBLE_DEVICES=$gpu_id python scripts/measure_comet.py $SAVE_PATH $pt $tgt &
            pids[$gpu_id]=$!
        fi
    done
    if [[ ${#lang_pairs[@]} -eq 0 ]]; then
        break
    fi
    sleep 10
done

wait


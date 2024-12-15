#!/bin/bash

EXPERIMENT_NAME=$1
EXPERIMENT_ID=$2
pt=$3
ROOT_PATH=$4

WORK_PATH=${ROOT_PATH}/ec_40_scripts
SAVE_PATH=${WORK_PATH}/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}

langs=(en bg da es uk hi ro de cs pt nl mr ur sv gu ar fr ru it pl he kn bn be mt am is sd)
n=5
num_langs=${#langs[@]}
codes_per_part=$(( (num_langs + n - 1) / n ))

for ((i=0; i<n; i++)); do
    start_index=$(( i * codes_per_part ))
    end_index=$(( start_index + codes_per_part ))
    if (( end_index > num_langs )); then
        end_index=$num_langs
    fi

    part_langs=("${langs[@]:start_index:end_index - start_index}")
    joined_langs=$(IFS=, ; echo "${part_langs[*]}")

    python evaluation/measure_offtarget.py $SAVE_PATH $pt $joined_langs &
done

wait

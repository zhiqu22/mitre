#!/bin/bash


EXPERIMENT_NAME=$1
EXPERIMENT_ID=$2
PT=${3}
num_gpus=${4}
ROOT_PATH=${5}
WORK_PATH=${ROOT_PATH}/ec_40_scripts
cd ${WORK_PATH}

languages=("en" "bg" "so" "ca" "da" "be" "bs" "mt" "es" "uk" \
           "am" "hi" "ro" "no" "ti" "de" "cs" "lb" "pt" "nl" \
           "mr" "is" "ne" "ur" "oc" "ast" "ha" "sv" "kab" "gu" \
           "ar" "fr" "ru" "it" "pl" "sr" "sd" "he" "af" "kn" "bn")


num_languages=${#languages[@]}
total_pairs=$((num_languages * (num_languages - 1)))

pairs_per_string=$((total_pairs / num_gpus))
remainder=$((total_pairs % num_gpus))

result=()
current_pairs=""
pair_count=0
string_count=0

for ((i=0; i<num_languages; i++)); do
    for ((j=0; j<num_languages; j++)); do
        if [[ $i -ne $j ]]; then
            pair="${languages[i]}-${languages[j]}"
            current_pairs+="${pair},"
            pair_count=$((pair_count + 1))

        if [[ $pair_count -eq $pairs_per_string && $string_count -lt $((num_gpus-1)) ]] || \
            [[ $pair_count -eq $((pairs_per_string + 1)) && $remainder -gt 0 ]]; then
            result+=("${current_pairs%,}")
            current_pairs=""
            pair_count=0
            string_count=$((string_count + 1))
            if [[ $remainder -gt 0 ]]; then
                remainder=$((remainder - 1))
            fi
          fi
        fi
    done
done

if [[ -n $current_pairs ]]; then
    result+=("${current_pairs%,}")
fi

if [ $EXPERIMENT_NAME == 'ec_40_register' ];then
  DIR='--user-dir '${ROOT_PATH}'/fairseq/models/mitre/ '
fi

mkdir ${WORK_PATH}/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}


for ((i=0; i<${#result[@]}; i++)); do
    echo ${result[i]}
    CUDA_VISIBLE_DEVICES=${i}, python ${ROOT_PATH}/fairseq/fairseq_cli/multilingual_generate.py ${WORK_PATH}/fairseq-data-bin-sharded/test --gen-subset test \
        $DIR \
        --lang-dict ${WORK_PATH}/dicts/languages.txt \
        --lang-pairs ${WORK_PATH}/dicts/pairs.txt \
        --path ${WORK_PATH}/checkpoints/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${PT} \
        --remove-bpe sentencepiece \
        --required-batch-size-multiple 1 \
        --encoder-langtok tgt \
        --task translation_multi_simple_epoch \
        --infer-dir-path ${WORK_PATH}/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${PT} \
        --infer-pairs ${result[i]} \
        --no-progress-bar \
        --batch-size 64 \
        --num-workers 2 \
        --beam 5 &
done

wait

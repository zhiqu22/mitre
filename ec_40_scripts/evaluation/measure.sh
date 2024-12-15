#!/bin/bash

EXPERIMENT_NAME=$1
EXPERIMENT_ID=$2
PT=$3
ROOT_PATH=$4

WORK_PATH=${ROOT_PATH}/ec_40_scripts

FLORES_BPE=${WORK_PATH}/dicts/flores200_sacrebleu_tokenizer_spm.model
SPM_ENCODER=${ROOT_PATH}/fairseq/scripts/spm_encode.py

cd ${WORK_PATH}

RESULT_PATH=$WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${PT}
COUNT_PATH=$WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/

languages=("en" "bg" "so" "ca" "da" "be" "bs" "mt" "es" "uk" \
           "am" "hi" "ro" "no" "ti" "de" "cs" "lb" "pt" "nl" \
           "mr" "is" "ne" "ur" "oc" "ast" "ha" "sv" "kab" "gu" \
           "ar" "fr" "ru" "it" "pl" "sr" "sd" "he" "af" "kn" "bn")
# languages=(en bg da es uk hi ro de cs pt nl mr ur sv gu ar fr ru it pl he kn bn be mt am is sd)

for src in "${languages[@]}"; do
    for tgt in "${languages[@]}"; do
        if [ $src == $tgt ]; then
            continue
        fi
        # hypothesis
        cat ${RESULT_PATH}/${src}-${tgt}.raw.txt | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > ${RESULT_PATH}/${src}-${tgt}.h
        cat ${RESULT_PATH}/${src}-${tgt}.h | sacremoses -l ${tgt} detokenize > ${RESULT_PATH}/${src}-${tgt}.detok.h
        python ${SPM_ENCODER} --model ${FLORES_BPE} --output_format=piece --inputs=${RESULT_PATH}/${src}-${tgt}.detok.h --outputs=${RESULT_PATH}/${src}-${tgt}.sp.h
        # reference
        cat ${RESULT_PATH}/${src}-${tgt}.raw.txt | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > ${RESULT_PATH}/${src}-${tgt}.r
        cat ${RESULT_PATH}/${src}-${tgt}.r | sacremoses -l ${tgt} detokenize > ${RESULT_PATH}/${src}-${tgt}.detok.r
        python ${SPM_ENCODER} --model ${FLORES_BPE} --output_format=piece --inputs=${RESULT_PATH}/${src}-${tgt}.detok.r --outputs=${RESULT_PATH}/${src}-${tgt}.sp.r

        # source
        cat ${RESULT_PATH}/${src}-${tgt}.raw.txt | grep -P "^S" | sort -t '-' -k2n | cut -f 2- | sed 's/__[a-zA-Z_]*__ //' > ${RESULT_PATH}/${src}-${tgt}.s
        cat ${RESULT_PATH}/${src}-${tgt}.s | sacremoses -l ${src} detokenize > ${RESULT_PATH}/${src}-${tgt}.detok.s

        output=$(sacrebleu $RESULT_PATH/$src-$tgt.detok.h -w 4 -m chrf --chrf-word-order 2 < $RESULT_PATH/$src-$tgt.detok.r)
        echo ${src}"-"${tgt} >> ${COUNT_PATH}/${PT}.chrf
        echo ${output} >> ${COUNT_PATH}/${PT}.chrf

        output=$(sacrebleu $RESULT_PATH/$src-$tgt.sp.h -w 4 -tok none --force < $RESULT_PATH/$src-$tgt.sp.r)
        echo ${src}"-"${tgt} >> ${COUNT_PATH}/${PT}.spbleu
        echo ${output} >> ${COUNT_PATH}/${PT}.spbleu
    done
done
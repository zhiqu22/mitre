#!/bin/bash

EXPERIMENT_NAME=$1
EXPERIMENT_ID=$2
PT=$3
ROOT_PATH=$4

WORK_PATH=${ROOT_PATH}/mitre

FLORES_BPE=${WORK_PATH}/dicts/flores200_sacrebleu_tokenizer_spm.model
SPM_ENCODER=${ROOT_PATH}/fairseq/scripts/spm_encode.py

cd ${WORK_PATH}

RESULT_PATH=$WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${PT}
COUNT_PATH=$WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/

IFS=" " read -r -a languages <<< $(echo $(python scripts/functions.py get_languages) | tr -d '[],' | tr -d "'")

for src in "${languages[@]}"; do
    for tgt in "${languages[@]}"; do
        if [ $src == $tgt ]; then
            continue
        fi
        # hypothesis
        cat ${RESULT_PATH}/${src}-${tgt}.raw.txt | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > ${RESULT_PATH}/${src}-${tgt}.h
        python ${SPM_ENCODER} --model ${FLORES_BPE} --output_format=piece --inputs=${RESULT_PATH}/${src}-${tgt}.h --outputs=${RESULT_PATH}/${src}-${tgt}.sp.h
        # # reference
        cat ${RESULT_PATH}/${src}-${tgt}.raw.txt | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > ${RESULT_PATH}/${src}-${tgt}.r
        python ${SPM_ENCODER} --model ${FLORES_BPE} --output_format=piece --inputs=${RESULT_PATH}/${src}-${tgt}.r --outputs=${RESULT_PATH}/${src}-${tgt}.sp.r

        # # source
        cat ${RESULT_PATH}/${src}-${tgt}.raw.txt | grep -P "^S" | sort -t '-' -k2n | cut -f 2- | sed 's/__[a-zA-Z_]*__ //' > ${RESULT_PATH}/${src}-${tgt}.s
        
        output=$(sacrebleu $RESULT_PATH/$src-$tgt.h -w 4 -m chrf --chrf-word-order 2 < $RESULT_PATH/$src-$tgt.r)
        echo ${src}"-"${tgt} >> ${COUNT_PATH}/${PT}.chrf
        echo ${output} >> ${COUNT_PATH}/${PT}.chrf

        output=$(sacrebleu $RESULT_PATH/$src-$tgt.sp.h -w 4 -tok none --force < $RESULT_PATH/$src-$tgt.sp.r)
        echo ${src}"-"${tgt} >> ${COUNT_PATH}/${PT}.spbleu
        echo ${output} >> ${COUNT_PATH}/${PT}.spbleu
    done
done



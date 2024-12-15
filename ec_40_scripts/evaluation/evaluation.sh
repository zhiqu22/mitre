#!/bin/bash

ROOT_PATH=$1
EXPERIMENT_NAME=$2
EXPERIMENT_ID=$3
PT_ID=$4

WORK_PATH=${ROOT_PATH}/ec_40_scripts
num_gpus=8
num_cpus=16
FLORES_BPE=${WORK_PATH}/dicts/flores200_sacrebleu_tokenizer_spm.model
SPM_ENCODER=${ROOT_PATH}/fairseq/scripts/spm_encode.py
AVERAGE=${ROOT_PATH}/fairseq/scripts/average_checkpoints.py

cd ${WORK_PATH}

mkdir $WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}
mkdir $WORK_PATH/tables/${EXPERIMENT_NAME}
mkdir $WORK_PATH/tables/${EXPERIMENT_NAME}/${EXPERIMENT_ID}

waiting_infer_pts=()
PT_DIR=$WORK_PATH/checkpoints/${EXPERIMENT_NAME}/${EXPERIMENT_ID}

if [[ "$PT_ID" == "all" ]] || [[ "$PT_ID" == "averaged" ]]; then
    for pt in ${PT_DIR}/checkpoint_*_*.pt; do
        pt=$(basename "$pt")
        waiting_infer_pts+=("$pt")
    done
    
else
    pt=${PT_DIR}/checkpoint_${PT_ID}.pt
    if [ -f "$pt" ]; then
        pt=$(basename "$pt")
        waiting_infer_pts+=("$pt")
    else
        echo "File $pt does not exist."
        exit 1
    fi
fi

# average last 5 checkpoints
if [[ "$PT_ID" == "all" ]] || [[ "$PT_ID" == "averaged" ]]; then
    sorted_checkpoints=$(for ckpt in "${waiting_infer_pts[@]}"; do
        y=$(echo $ckpt | awk -F'[_.]' '{print $(NF-1)}')
        echo "$y $ckpt"
    done | sort -nr | head -n 5 | awk '{print $2}' | while read -r filename; do
        echo "${PT_DIR}/${filename}"
    done | tr '\n' ' ')
    python3 ${AVERAGE} --inputs $sorted_checkpoints --output $PT_DIR/checkpoint_1_averaged.pt
    waiting_infer_pts+=("checkpoint_1_averaged.pt")
fi

for pt in "${waiting_infer_pts[@]}"; do
    if [[ "$PT_ID" == "averaged" ]] && [[ "$pt" != "checkpoint_1_averaged.pt" ]]; then
        continue
    fi
    mkdir $WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${pt}
    bash evaluation/inference.sh ${EXPERIMENT_NAME} ${EXPERIMENT_ID} ${pt} ${num_gpus} ${ROOT_PATH}
    bash evaluation/measure.sh ${EXPERIMENT_NAME} ${EXPERIMENT_ID} ${pt} ${ROOT_PATH}
    bash evaluation/call_offtarget.sh ${EXPERIMENT_NAME} ${EXPERIMENT_ID} ${pt} ${ROOT_PATH}
    bash evaluation/call_comet.sh ${EXPERIMENT_NAME} ${EXPERIMENT_ID} ${pt} ${ROOT_PATH} ${num_gpus}
    python evaluation/make_table.py ${WORK_PATH} ${EXPERIMENT_NAME} ${EXPERIMENT_ID} ${pt}

    rm $WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${pt}/*.h
    rm $WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${pt}/*.r
    rm $WORK_PATH/results/${EXPERIMENT_NAME}/${EXPERIMENT_ID}/${pt}/*.s
done
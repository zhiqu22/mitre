#!/bin/bash

EXPERIMENT_ID=1
EXPERIMENT_NAME=ec_40_register

ROOT_PATH=$1
WORK_PATH=${ROOT_PATH}/ec_40_scripts
FAIRSEQ_PATH=${ROOT_PATH}/fairseq
SAVE_PATH=${WORK_PATH}/checkpoints/${EXPERIMENT_NAME}/
RESULTS_PATH=${WORK_PATH}/results/${EXPERIMENT_NAME}/
LOG_PATH=${WORK_PATH}/logs/${EXPERIMENT_NAME}/
BIN_PATH=${WORK_PATH}/fairseq-data-bin-sharded/shard

FACTOR=1

mkdir ${WORK_PATH}/results
mkdir ${WORK_PATH}/checkpoints
mkdir ${WORK_PATH}/logs
mkdir $SAVE_PATH
mkdir $LOG_PATH
mkdir $RESULTS_PATH

cd ${FAIRSEQ_PATH}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ${BIN_PATH}0:${BIN_PATH}1:${BIN_PATH}2:${BIN_PATH}3:${BIN_PATH}4 \
    --user-dir models/mitre/ \
    --lang-dict ${WORK_PATH}/dicts/languages.txt \
    --lang-pairs ${WORK_PATH}/dicts/pairs.txt \
    --task translation_multi_simple_epoch \
    --encoder-langtok tgt \
    --arch transformer_register_big --register-factor ${FACTOR} \
    --memory-efficient-fp16 \
    --decoder-normalize-before \
    --encoder-layers 0 --decoder-layers 12 \
    --sampling-method temperature --sampling-temperature 5 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 --update-freq 8 --max-update 200000 \
    --share-all-embeddings \
    --max-source-positions 256 --max-target-positions 256 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --seed 1234 --patience -1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --weight-decay 0.0 \
    --dropout 0.1 --attention-dropout 0.1 \
    --ddp-backend no_c10d \
    --save-dir ${SAVE_PATH}/${EXPERIMENT_ID} \
    --save-interval-updates 10000 --keep-interval-updates 5 --no-epoch-checkpoints --log-interval 100 > ${LOG_PATH}/${EXPERIMENT_ID}.log
    
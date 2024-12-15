#!/bin/bash

EXPERIMENT_ID=1
EXPERIMENT_NAME=mitre

MASTER_NODE="localhost"
# nodes in total
NUM_NODES=1
# current node, begin from 0
RANK=0
PORT=12345
GPU_PER_NODE=8

ROOT_PATH=$1
WORK_PATH=${ROOT_PATH}/MITRE
FAIRSEQ_PATH=${ROOT_PATH}/fairseq
SAVE_PATH=${WORK_PATH}/checkpoints/${EXPERIMENT_NAME}/
RESULTS_PATH=${WORK_PATH}/results/${EXPERIMENT_NAME}/
LOG_PATH=${WORK_PATH}/logs/${EXPERIMENT_NAME}/

BIN_PATH=""
for i in {00..99};do
    BIN_PATH=${BIN_PATH}:${WORK_PATH}/mitre-bin/shard-${i}
done
BIN_PATH=${BIN_PATH:1}
echo ${BIN_PATH}

cd ${FAIRSEQ_PATH}

mkdir ${WORK_PATH}/checkpoints
mkdir ${WORK_PATH}/results
mkdir ${WORK_PATH}/logs
mkdir $SAVE_PATH
mkdir $LOG_PATH
mkdir $RESULTS_PATH

# 400M model is transformer_register_big
# 900M model is transformer_register_massive
torchrun --nproc_per_node=${GPU_PER_NODE} \
--nnodes=${NUM_NODES} \
--node_rank=${RANK} \
--MASTER_NODE_addr=${MASTER_NODE} \
--MASTER_NODE_port=${PORT} \
${FAIRSEQ_PATH}/train.py ${BIN_PATH} \
--user-dir models/mitre/ \
--lang-dict ${WORK_PATH}/dicts/mitre_languages.txt \
--lang-pairs ${WORK_PATH}/dicts/mitre_pairs.txt \
--arch transformer_register_big --gist-factor 1 \
--task translation_multi_simple_epoch \
--encoder-langtok tgt \
--memory-efficient-fp16 \
--ddp-backend no_c10d \
--seed 42 \
--patience -1 \
--decoder-normalize-before \
--encoder-layers 0 --decoder-layers 24 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --weight-decay 0.0 \
--share-all-embeddings \
--max-source-positions 256 --max-target-positions 256 \
--skip-invalid-size-inputs-valid-test \
--sampling-method temperature --sampling-temperature 1.0 \
--max-tokens 1408 \
--update-freq 10 \
--max-update 500000 \
--lr 0.002 \
--lr-scheduler inverse_sqrt \
--warmup-updates 8000 \
--warmup-init-lr 1e-07 \
--stop-min-lr 1e-09 \
--dropout 0.1 --attention-dropout 0.1 \
--save-dir ${SAVE_PATH}/${EXPERIMENT_ID} \
--validate-interval-updates 10000 \
--save-interval-updates 10000 --keep-interval-updates 20 --no-epoch-checkpoints \
--log-interval 100 \
--log-format simple > ${LOG_PATH}/${EXPERIMENT_ID}_${RANK}.log &
wait
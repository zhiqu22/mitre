#!/bin/bash

ROOT_PATH=$1
WORK_PATH=${ROOT_PATH}/MITRE
FAIRSEQ_PATH=${ROOT_PATH}/fairseq

SPM_TRAIN=${FAIRSEQ_PATH}/scripts/spm_train.py
SPM_ENCODE=${FAIRSEQ_PATH}/scripts/spm_encode.py
LEN_CLEAN=${ROOT_PATH}/mosesdecoder/scripts/training/clean-corpus-n.perl

SPM_MODEL=${WORK_PATH}/dicts/mitre_spm.model
BIN_DICT=${WORK_PATH}/dicts/mitre_dict.txt

# sharding bpe and binary data 
n_shards=100
min_lines_per_file=50000
# params for handing splited data in sharding bpe and binary data
denominator=1
nominator=1

cd ${WORK_PATH}
mkdir logs
mkdir logs/preprocess
mkdir results
mkdir checkpoints
mkdir tables
mkdir dicts
mkdir bpe
mkdir bpe_valid
mkdir bpe_test
mkdir mitre-bin
mkdir valid_bin
mkdir test_bin

mkdir raw_data/zips
mkdir raw_data/unzips
mkdir raw_data/depunc
mkdir raw_data/dedup
mkdir raw_data/clean

pip install zipfile

python scripts/download_raw_data.py
python scripts/unzip_raw_data.py
rm raw_data/unzips/*/*.scores
rm raw_data/unzips/*/LICENSE
rm raw_data/unzips/*/README
rm -r raw_data/zips

curl -o raw_data/histograms.tar.gz https://dl.fbaipublicfiles.com/m2m_100/histograms.tar.gz 
tar -xvzf raw_data/histograms.tar.gz -C raw_data/
mv raw_data/checkpoint/edunov/cc60_multilingual/clean_hists raw_data/clean_hists
rm -r raw_data/checkpoint/
rm raw_data/histograms.tar.gz

pairs=$(find raw_data/unzips -mindepth 1 -maxdepth 1 -type d -printf '%f\n')
for pair in ${pairs}; do
    echo processing ${pair}
    src=$(echo $pair | cut -d '-' -f 1)
    tgt=$(echo $pair | cut -d '-' -f 2)
    python scripts/remove_punctuations.py \
      --src-file raw_data/unzips/${src}-${tgt}/${src}-${tgt}.${src} \
      --tgt-file raw_data/unzips/${src}-${tgt}/${src}-${tgt}.${tgt} \
      --bitext raw_data/depunc/${src}-${tgt}.depunc \
      --src-lang ${src} \
      --tgt-lang ${tgt}
    paste raw_data/depunc/${src}-${tgt}.depunc.${src} \
          raw_data/depunc/${src}-${tgt}.depunc.${tgt}| awk '!x[$0]++' > raw_data/dedup/${src}-${tgt}.dedup
    sort -R raw_data/dedup/${src}-${tgt}.dedup > raw_data/dedup/${src}-${tgt}.shuffle.dedup
    cut -f1 raw_data/dedup/${src}-${tgt}.shuffle.dedup > raw_data/dedup/${src}-${tgt}.dedup.$src
    cut -f2 raw_data/dedup/${src}-${tgt}.shuffle.dedup > raw_data/dedup/${src}-${tgt}.dedup.$tgt
    python scripts/clean_histogram.py \
      --src $src --tgt $tgt \
      --src-file raw_data/dedup/${src}-${tgt}.dedup.$src --tgt-file raw_data/dedup/${src}-${tgt}.dedup.$tgt \
      --src-output-file raw_data/clean/${src}-${tgt}.${src}  --tgt-output-file raw_data/clean/${src}-${tgt}.${tgt} \
      --histograms raw_data/clean_hists
done
rm -r raw_data/unzips
rm -r raw_data/dedup
rm -r raw_data/depunc


# SentencePiece Model and Vocabulary is provided in dicts/
# if [[ ! -e ${SPM_MODEL} ]]; then
#     BPE_FILES=raw_data/clean/en-de.en
#     IFS=" " read -r -a languages <<< $(echo $(python scripts/functions.py get_languages) | tr -d '[],' | tr -d "'")
#     for language in "${languages[@]}"; do
#         if [[ ${language} == 'en' ]]; then
#             continue
#         fi
#         file_path=raw_data/clean/en-${language}.${language}
#         BPE_FILES=${BPE_FILES}","${file_path}
#     done
#     echo $BPE_FILES
#     python $SPM_TRAIN --input=$BPE_FILES \
#         --model_prefix=dicts/mitre_spm \
#         --vocab_size=160000 \
#         --character_coverage=1.0 \
#         --input_sentence_size=150000000 \
#         --shuffle_input_sentence=True \
#         --train_extremely_large_corpus=True \
#         --num_threads=32
#     cut -f 1 dicts/mitre_spm.vocab | tail -n +4 | sed "s/$/ 1/g" > ${BIN_DICT}
# fi


# bpe and binary valid set
IFS=" " read -r -a pairs <<< $(echo $(python scripts/functions.py get_bidirection_pairs) | tr -d '[],' | tr -d "'")
for pair in "${pairs[@]}"; do
    src=$(echo $pair | cut -d'-' -f1)
    tgt=$(echo $pair | cut -d'-' -f2)
    flores_src=$(python scripts/functions.py translate_language_code ${src} m2m flores)
    flores_tgt=$(python scripts/functions.py translate_language_code ${tgt} m2m flores)

    mkdir bpe_valid/${src}-${tgt}

    python ${SPM_ENCODE} --model ${SPM_MODEL} --output_format=piece \
        --inputs=raw_data/floresp-v2.0-rc.3/dev/dev.${flores_src} \
        --outputs=bpe_valid/${src}-${tgt}/valid.${src}

    python ${SPM_ENCODE} --model ${SPM_MODEL} --output_format=piece \
        --inputs=raw_data/floresp-v2.0-rc.3/dev/dev.${flores_tgt} \
        --outputs=bpe_valid/${src}-${tgt}/valid.${tgt}
    fairseq-preprocess \
      --source-lang ${src} --target-lang ${tgt} \
      --validpref bpe_valid/${src}-${tgt}/valid \
      --destdir valid_bin \
      --workers 4 \
      --srcdict ${BIN_DICT} --tgtdict ${BIN_DICT}
done

# bpe and binary test set
IFS=" " read -r -a languages <<< $(echo $(python scripts/functions.py get_languages) | tr -d '[],' | tr -d "'")
for src in "${languages[@]}"; do
    for tgt in "${languages[@]}"; do
        if [[ ${src} == ${tgt} ]];then
            continue
        fi
        flores_src=$(python scripts/functions.py translate_language_code ${src} m2m flores)
        flores_tgt=$(python scripts/functions.py translate_language_code ${tgt} m2m flores)

        mkdir bpe_test/${src}-${tgt}
        
        python ${SPM_ENCODE} --model ${SPM_MODEL} --output_format=piece \
            --inputs=raw_data/floresp-v2.0-rc.3/devtest/devtest.${flores_src} \
            --outputs=bpe_test/${src}-${tgt}/test.${src}

        python ${SPM_ENCODE} --model ${SPM_MODEL} --output_format=piece \
            --inputs=raw_data/floresp-v2.0-rc.3/devtest/devtest.${flores_tgt} \
            --outputs=bpe_test/${src}-${tgt}/test.${tgt}
        fairseq-preprocess \
            --source-lang ${src} --target-lang ${tgt} \
            --testpref bpe_test/${src}-${tgt}/test \
            --destdir test_bin \
            --workers 4 \
            --srcdict ${BIN_DICT} --tgtdict ${BIN_DICT}
    done
done

rm valid_bin/dict.*
rm valid_bin/preprocess.log 
for i in {00..99};do
    if [ -e mitre-bin/shard-${i} ]; then
        break
    fi
    mkdir mitre-bin/shard-${i}
    cp valid_bin/* mitre-bin/shard-${i}/
done

IFS=" " read -r -a pairs <<< $(echo $(python scripts/functions.py get_partial_pairs ${denominator} ${nominator}) | tr -d '[],' | tr -d "'")
for pair in "${pairs[@]}"; do
    src=$(echo $pair | cut -d '-' -f 1)
    tgt=$(echo $pair | cut -d '-' -f 2)
    dir_path=bpe/${src}-${tgt}
    echo bpe ${src}-${tgt}
    mkdir ${dir_path}
    python ${SPM_ENCODE} --model ${SPM_MODEL} --output_format=piece \
        --inputs=raw_data/clean/${src}-${tgt}.${src} \
        --outputs=bpe/${src}-${tgt}/train.unclean.${src}
    python ${SPM_ENCODE} --model ${SPM_MODEL} --output_format=piece \
        --inputs=raw_data/clean/${src}-${tgt}.${tgt} \
        --outputs=bpe/${src}-${tgt}/train.unclean.${tgt}
    perl ${LEN_CLEAN} --ratio 3 bpe/${src}-${tgt}/train.unclean $src $tgt bpe/${src}-${tgt}/train 1 250
    rm bpe/${src}-${tgt}/train.unclean*
   
    src_lines=$(wc -l < bpe/${src}-${tgt}/train.${src})
    tgt_lines=$(wc -l < bpe/${src}-${tgt}/train.${tgt})
    if [ "$src_lines" -eq "$tgt_lines" ]; then
        echo ${src}-${tgt} has ${src_lines} sentences >> logs/preprocess/check_length.log
    else
        echo ${src}-${tgt} has different sentences: ${src_lines} and ${tgt_lines} >> logs/preprocess/check_length.log
    fi
done

for pair in "${pairs[@]}"; do
    src=$(echo $pair | cut -d '-' -f 1)
    tgt=$(echo $pair | cut -d '-' -f 2)
    echo binarizing ${src}-${tgt}

    mkdir -p bpe/${src}-${tgt}-tmp
    for i in $(seq -w 0 $counter_shards); do
        mkdir bpe/${src}-${tgt}-shard-${i}
    done

    total_lines=$(wc -l < bpe/${src}-${tgt}/train.${src})

    lines_per_shard=$((total_lines / n_shards))
    echo ${total_lines}

    if [ "$lines_per_shard" -ge "$min_lines_per_file" ]; then
        actual_n=$((n_shards - 1))
    else
        actual_n=$((total_lines / min_lines_per_file))
        lines_per_shard=$min_lines_per_file
    fi
    echo ${actual_n}
    echo ${lines_per_shard}

    split -l $((lines_per_shard + 1)) -d -a 2 bpe/${src}-${tgt}/train.${src} bpe/${src}-${tgt}-tmp/train.${src}.shard-
    split -l $((lines_per_shard + 1)) -d -a 2 bpe/${src}-${tgt}/train.${tgt} bpe/${src}-${tgt}-tmp/train.${tgt}.shard-
    
    num_iter=$(((n_shards / actual_n) + 3))
    counter=0
    for ((i=1; i<${num_iter}; i++)); do
        if [ ${counter} -ge ${counter_shards} ]; then
            break
        fi
        for j in $(seq 00 ${actual_n}); do
            formatted_counter=$(printf "%02d" $counter)
            formatted_j=$(printf "%02d" $j)
            cp bpe/${src}-${tgt}-tmp/train.${src}.shard-${formatted_j} bpe/${src}-${tgt}-shard-${formatted_counter}/train.${src}
            cp bpe/${src}-${tgt}-tmp/train.${tgt}.shard-${formatted_j} bpe/${src}-${tgt}-shard-${formatted_counter}/train.${tgt}
            fairseq-preprocess --dataset-impl mmap \
            --source-lang ${src} \
            --target-lang ${tgt} \
            --trainpref bpe/${src}-${tgt}-shard-${formatted_counter}/train \
            --destdir mitre-bin/shard-${formatted_counter} \
            --thresholdtgt 0 \
            --thresholdsrc 0 \
            --workers 16 \
            --srcdict ${BIN_DICT} \
            --tgtdict ${BIN_DICT}
            rm -r bpe/${src}-${tgt}-shard-${formatted_counter}
            if [ ${counter} -ge ${counter_shards} ]; then
                break
            fi
            ((counter++))
        done
    done
    rm -r bpe/${src}-${tgt}-tmp
done

# has already been provided in dicts/
# IFS=" " read -r -a pairs <<< $(echo $(python scripts/functions.py get_bidirection_pairs) | tr -d '[],' | tr -d "'")
# current_pairs=""
# for pair in "${pairs[@]}"; do
#     current_pairs+=${pair},
# done
# current_pairs=${current_pairs::-1}
# echo $current_pairs > dicts/mitre_pairs.txt

# IFS=" " read -r -a languages <<< $(echo $(python scripts/functions.py get_languages) | tr -d '[],' | tr -d "'")
# for language in "${languages[@]}"; do
#     echo $language >> dicts/mitre_languages.txt
# done
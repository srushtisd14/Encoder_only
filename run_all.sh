#!/bin/bash

set -e

MODELS=(
    bert-base
    roberta-base
    distilbert
    albert-base-v2
    electra-base
    deberta-v3-base
    mpnet-base
)

DATASETS=(
    sst2
    imdb
    ag_news
    mrpc
    qnli
    mnli
)

for MODEL in "${MODELS[@]}"
do
    for DATASET in "${DATASETS[@]}"
    do

        echo "=========================================="
        echo "Running $MODEL on $DATASET"
        echo "=========================================="

        python3 run_experiments.py \
            --model "$MODEL" \
            --dataset "$DATASET"

        echo "SUCCESS: $MODEL - $DATASET"
        echo "=========================================="

    done
done
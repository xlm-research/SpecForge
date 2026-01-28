#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B
NUM_GPUS=${1:-4}
TP_SIZE=${2:-2}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

target_model_path="/mnt/lustre/share_data/models/Qwen/Qwen3-30B-A3B"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $target_model_path \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
    --train-data-path $ROOT_DIR/datasets/allava_laion/yb_2001_instruct.json \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-30b-a3b-instruct-eagle3-allava-laion \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 32768 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --sp-ulysses-size 2 \
    --attention-backend usp \
    --target-model-backend sglang 

    # --sp_ring_size 2 \
    # --sglang-enable-dp-attention \
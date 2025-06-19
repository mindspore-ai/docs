#!/bin/bash

# Runs Mixtral 8x7B model
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

LOAD_PATH="/path/to/checkpoints"
TOKENIZER_MODEL="/path/to/tokenizer.json"
DATA_PATH="/path/to/wiki_text_document"

TP=1
PP=4
EP=1

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 163840
    --num-layers 4
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 8
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --no-rope-fusion
    --swiglu
    --untie-embeddings-and-output-weights
    --num-query-groups 8
    --no-masked-softmax-fusion
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.3
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-pos-emb-head-dim 64
    --v-head-dim 192
    --qk-head-dim 128
    --qk-layernorm
    --vocab-size 129280
    --make-vocab-size-divisible-by 129280
    --use-flash-attn
    --multi-latent-attention
    --attention-backend flash
)

MOE_ARGS=(
    --moe-layer-freq '[0]+[1]*3'
    --num-experts 16
    --moe-router-topk 8
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-shared-expert-intermediate-size 2048
    --moe-ffn-hidden-size 2048
    --moe-router-group-topk 0
    --moe-router-topk-scaling-factor 1.5
    --moe-router-score-function sigmoid
    --moe-router-dtype fp32
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 1,0,0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4
    --train-iters 1000
    --lr 1.e-6
    --lr-decay-style constant
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --bf16
    --finetune
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1000 \
    --no-load-optim \
    --no-load-rng \
    --ckpt-format torch \
    --load $LOAD_PATH
)

logtime=$(date +%Y%m%d)_$(date +%H%M%S)
torchrun ${DISTRIBUTED_ARGS[@]} /path/to/Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} 2>&1 | tee logs/${logtime}.log
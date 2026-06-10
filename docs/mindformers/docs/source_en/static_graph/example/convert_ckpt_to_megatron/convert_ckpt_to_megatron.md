<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} Deprecated
:class: warning

This page belongs to the **Static Graph (GRAPH_MODE) Implementation** section and has been marked as deprecated. New features are primarily being developed in the "r2.0.0 Dynamic Graph Implementation" section. Please refer to the dynamic graph documentation first.
```

# Practical Case: Converting Model Weights to Megatron Model Weights

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/static_graph/example/convert_ckpt_to_megatron/convert_ckpt_to_megatron.md)

This case provides a method for converting the model weights (in Safetensors format) of the [MindSpore Transformers](https://atomgit.com/mindspore/mindformers) library to the format used in the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) library to facilitate accuracy comparison or migration training. After conversion, Megatron-LM weights are of the BF16 type.

## Environment Preparations

### Preparing Code

1. Clone the Megatron-LM code repository and switch to the **core_r0.12.0** branch.

    ```shell
    git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.12.0
    ```

2. Copy the [conversion script](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/convert_ckpt_to_megatron/convert_ckpt_to_megatron/loader_core_mf.py) to the **Megatron-LM/tools/checkpoint/** directory.

## Model Weight Preparations

Convert the Safetensors weights saved by MindSpore Transformers.

> - Currently, only the weights of GPT-like models (such as GPT and Qwen) composed of SelfAttention and MLP can be converted. MLA and MoE are not supported.
> - Only complete weights that are not split for distribution are supported. If weights are distributed, merge them by referring to [Weight Merging](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html#weight-merging).

## Weight Conversion Procedure

1. Go to the Megatron-LM directory.

    ```shell
    cd Megatron-LM
    ```

2. Run the weight conversion commands (with the actual path and parameters).

    ```shell
    TARGET_TP_SIZE=2  # Target tensor parallelism (TP).
    TARGET_PP_SIZE=2  # Target pipeline parallelism (PP).

    python ./tools/checkpoint/convert.py \
        --megatron-path 'path_to_megatron' \
        --model-type GPT \
        --loader core_mf \
        --saver core \
        --target-tensor-parallel-size ${TARGET_TP_SIZE} \
        --target-pipeline-parallel-size ${TARGET_PP_SIZE} \
        --load-dir "path_to_ms_ckpt" \
        --save-dir "path_to_megatron_ckpt" \
        --loader-transformer-impl local \
        --saver-transformer-impl local \
        --position-embedding-type "rope" \
        --true-vocab-size 128000  \
        --padded-vocab-size 128000  \
        --num-layers 32 \
        --seq-length 2048 \
        --hidden-size 4096 \
        --ffn-hidden-size 16384 \
        --num-attention-heads 32 \
        --num-query-groups 16 \
        --normalization "RMSNorm" \
        --add-bias-linear \
        --swiglu
    ```

    Parameters

    | Name| Required| Default Value| Description|
    | ---- | --------- | ------ | -------- |
    | `--megatron-path` | Yes| None| Root directory of the Megatron-LM repository.|
    | `--model-type` | Yes| None| Model type (for example, **GPT**).|
    | `--loader` | Yes| None| Loader type (**core_mf** in this example).|
    | `--saver` | Yes| None| Saver type (for example, **core**).|
    | `--target-tensor-parallel-size` | Yes| None| Target TP.|
    | `--target-pipeline-parallel-size` | Yes| None| Target PP.|
    | `--load-dir` | Yes| None| Path to the Safetensors weight files exported from MindSpore. (It may include only one file or folder.)|
    | `--save-dir` | Yes| None| Megatron weight output directory.|
    | `--loader-transformer-impl` | No| transformer_engine | Transformer implementation of the loader, which can be **local** (for precision comparison) or **transformer_engine**.|
    | `--saver-transformer-impl` | No| transformer_engine | Transformer implementation of the saver, which can be **local** (for precision comparison) or **transformer_engine**.|
    | `--position-embedding-type` | No| learned_absolute | Position encoding type (**learned_absolute** or **rope**).|
    | `--true-vocab-size` | No| None | Actual vocabulary size of the model. If this parameter is specified, the padding of the embedding table is removed.|
    | `--padded-vocab-size` | No| 128000 | Vocabulary size after padding. In MindSpore Transformers, the value is usually the same as the actual vocabulary size.|
    | `--num-layers` | No| 512 | Number of transformer layers.|
    | `--seq-length` | No| 2048 | Maximum sequence length.|
    | `--hidden-size` | No| 512 | Dimension of the hidden layer.|
    | `--ffn-hidden-size` | No| 128 | Dimension of the hidden layer in the feedforward network.|
    | `--num-attention-heads` | No| 64 | Number of attention heads.|
    | `--num-query-groups` | No| None | Number of query groups.|
    | `--normalization` | No| RMSNorm | Normalization type.|
    | `--add-bias-linear` | No| False | Specifies whether to add bias to the linear layer. (The value is of the Boolean type. Set the parameter to **True** if you need to add bias.)|
    | `--swiglu` | No| False | Specifies whether to activate SwiGLU. (The value is of the Boolean type. Set the parameter to **True** if you need to activate SwiGLU.)|
    | `--ms2torch-ckpt-path` | No| ./ms2pt_checkpoint | Path to the intermediate weights output during conversion.|

3. After a successful operation, ensure that weights are saved to the location (`./ms2pt_checkpoint` by default) specified by `--ms2torch-ckpt-path`.

## FAQ

- **Q: What can I do if an error is reported when Megatron loads converted weights?**

  A: Ensure that all model structure parameters (such as the number of layers, hidden layer dimension, and vocabulary size) are the same as those of the original model.

- **Q: Are the MoE and other structures supported?**

  A: Currently, only the standard SelfAttention+MLP structure is supported.

- **Q: Are distributed weights supported?**

  A: No. You need to merge the weights first.

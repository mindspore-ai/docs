# 转换模型权重为Megatron模型权重的实践案例

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/convert_ckpt_to_megatron/convert_ckpt_to_megatron.md)

本案例提供了一个将 [MindSpore Transformers](https://gitee.com/mindspore/mindformers) 库的模型权重（safetensors格式）转换为 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 库的模型权重格式的方法，以便后续进行精度比对或迁移训练。转换后的 Megatron-LM 权重为bf16类型。

## 环境准备

### 代码准备

1. 克隆Megatron-LM代码仓库，并切换到 core_r0.12.0 分支：

    ```shell
    git clone https://github.com/NVIDIA/Megatron-LM.git -b core_r0.12.0
    ```

2. 拷贝[转换脚本](https://gitee.com/mindspore/docs/tree/master/docs/mindformers/docs/source_zh_cn/example/convert_ckpt_to_megatron/convert_ckpt_to_megatron/loader_core_mf.py)到 Megatron-LM/tools/checkpoint/ 目录下。

## 模型权重准备

使用 MindSpore Transformers 保存的safetensors权重进行转换。

> - 当前仅支持由SelfAttention和MLP组成的类GPT模型权重转换（如GPT、Qwen等），暂不支持MLA和MoE。
> - 仅支持未分布式切分的完整权重。如为分布式权重，请先参考[权重合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html#%E6%9D%83%E9%87%8D%E5%90%88%E5%B9%B6)进行合并。

## 权重转换步骤

1. 进入 Megatron-LM 目录：

    ```shell
    cd Megatron-LM
    ```

2. 执行权重转换命令（请根据实际路径和参数填写）：

    ```shell
    TARGET_TP_SIZE=2  # 目标张量并行数
    TARGET_PP_SIZE=2  # 目标流水线并行数

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

    参数说明

    | 名称 | 可选/必选 | 默认值 | 功能介绍 |
    | ---- | --------- | ------ | -------- |
    | `--megatron-path` | 必选 | 无 | Megatron-LM仓库的根目录路径 |
    | `--model-type` | 必选 | 无 | 模型类型（如GPT） |
    | `--loader` | 必选 | 无 | 加载器类型（此处为core_mf） |
    | `--saver` | 必选 | 无 | 保存器类型（如core） |
    | `--target-tensor-parallel-size` | 必选 | 无 | 目标张量并行数（TP） |
    | `--target-pipeline-parallel-size` | 必选 | 无 | 目标流水线并行数（PP） |
    | `--load-dir` | 必选 | 无 | MindSpore导出的safetensors权重文件路径（单文件或文件夹） |
    | `--save-dir` | 必选 | 无 | Megatron权重输出目录 |
    | `--loader-transformer-impl` | 可选 | transformer_engine | 加载器Transformer实现，local或transformer_engine，用于精度比对时，选择local |
    | `--saver-transformer-impl` | 可选 | transformer_engine | 保存器Transformer实现，local或transformer_engine，用于精度比对时，选择local |
    | `--position-embedding-type` | 可选 | learned_absolute | 位置编码类型（learned_absolute或rope） |
    | `--true-vocab-size` | 可选 | None | 模型实际词表大小，指定时会去除embedding表padding |
    | `--padded-vocab-size` | 可选 | 128000 | pad后的词表大小，MindSpore Transformers 中一般与实际词表相同 |
    | `--num-layers` | 可选 | 512 | Transformer层数 |
    | `--seq-length` | 可选 | 2048 | 最大序列长度 |
    | `--hidden-size` | 可选 | 512 | 隐藏层维度 |
    | `--ffn-hidden-size` | 可选 | 128 | 前馈网络隐藏层维度 |
    | `--num-attention-heads` | 可选 | 64 | 注意力头数 |
    | `--num-query-groups` | 可选 | None | Query分组数 |
    | `--normalization` | 可选 | RMSNorm | 归一化类型 |
    | `--add-bias-linear` | 可选 | False | 为线性层添加bias（布尔开关，添加该参数则为True） |
    | `--swiglu` | 可选 | False | 使用SwiGLU激活（布尔开关，添加该参数则为True） |
    | `--ms2torch-ckpt-path` | 可选 | ./ms2pt_checkpoint | 中间转换权重的输出路径 |

3. 执行成功，权重保存在`--ms2torch-ckpt-path`配置的位置，默认在`./ms2pt_checkpoint`位置。

## 常见问题

- **Q: 权重转换后Megatron加载报错，怎么办？**  
  A: 请确保所有模型结构参数（如层数、隐藏维度、词表大小等）与原始模型完全一致。

- **Q: 支持MoE或其他结构吗？**  
  A: 暂不支持，仅支持标准SelfAttention+MLP结构。

- **Q: 支持分布式权重吗？**  
  A: 暂不支持，请先合并权重。

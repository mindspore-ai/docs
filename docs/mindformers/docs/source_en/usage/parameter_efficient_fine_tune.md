# Parameter-Efficient Fine-Tuning (PEFT)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindformers/docs/source_en/usage/parameter_efficient_fine_tune.md)

## Overview

In the fine-tuning process of a deep learning model, all weights of the model need to be updated, which causes a large amount of computing resource consumption. Low-rank adaptation (LoRA) is a technology that significantly reduces the number of parameters required for fine-tuning by decomposing a partial weight matrix of a model into low-rank matrices. With Huawei Ascend AI processors, MindSpore deep learning framework, and MindFormers foundation model suite, LoRA can be used for PEFT of large-scale pretrained models (such as Llama2), providing efficient and flexible model customization capabilities.

## LoRA Principles

LoRA achieves a significant reduction in the number of parameters by decomposing the weight matrix of the original model into two low-rank matrices. For example, assuming that the size of a weight matrix **W** is *m* x *n*, the matrix is decomposed into two low-rank matrices **A** and **B** by using LoRA, where the size of **A** is *m* x *r*, and the size of **B** is *r* x *n* (*r* is far less than *m* and *n*). In the fine-tuning process, only the two low-rank matrices are updated without changing other parts of the original model.
This method not only greatly reduces the computing overhead of fine-tuning, but also retains the original performance of the model. It is especially suitable for model optimization in environments with limited data volume and computing resources. For details, see [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

## LoRA Fine-Tuning Process

The following key steps are required for LoRA PEFT:

1. **Pretrained model weight loading**: Load base weights from the pretrained model. These weights are parameters obtained after the model is trained on a large-scale dataset.

2. **Dataset preparation**: Select and prepare a dataset for fine-tuning. The dataset must be related to the target task, and the format must match the input format of the model.

3. **Fine-tuning parameter settings**: Set fine-tuning parameters, including the learning rate, optimizer type, and batch_size.

4. **LoRA parameter settings**: Set the **pet_config** parameter at the key layer (such as the attention layer) of the model and adjust the low-rank matrix to update the model parameters.

5. **Fine-tuning process startup**: Use the set parameters and datasets to start the fine-tuning process in a distributed environment.

6. **Evaluation and saving**: During or after fine-tuning, evaluate the model performance, and save the fine-tuned model weight.

## Using MindFormers for LoRA PEFT of Llama2

In the distributed environment of Ascend AI processors, the MindFormers suite can be used to easily implement the LoRA PEFT. The following shows the core configuration part of the LoRA fine-tuning of the Llama2 model and details the `pet_config` parameters.

### YAML File Example

For details about the complete YAML file, see [the Llama2 LoRA fine-tuning YAML file](https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/lora_llama2_7b.yaml).

```yaml
# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1
    seq_length: 4096
    hidden_size: 4096
    num_layers: 32
    num_heads: 32
    vocab_size: 32000
    compute_dtype: "float16"
    pet_config:
      pet_type: lora
      lora_rank: 16
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo'
  arch:
    type: LlamaForCausalLM
```

### pet_config Parameters

In **model_config**, **pet_config** is the core setting part of LoRA fine-tuning and is used to specify LoRA parameters. The parameters are described as follows:

- **pet_type**: specifies that the type of the parameter-efficient tuning (PET) is LoRA. The LoRA module is inserted in the key layer of the model to reduce the number of parameters required for fine-tuning.
- **lora_rank**: specifies the rank value of a low-rank matrix. A smaller rank value indicates fewer parameters that need to be updated during fine-tuning, reducing occupation of computing resources. The value **16** is a common equilibrium point, which significantly reduces the number of parameters while maintaining the model performance.
- **lora_alpha**: specifies the scaling ratio for weight update in the LoRA module. This value determines the amplitude and impact of weight update during fine-tuning. The value **16** indicates that the scaling amplitude is moderate, stabilizing the training process.
- **lora_dropout**: specifies the dropout probability in the LoRA module. Dropout is a regularization technique used to reduce overfitting risks. The value **0.05** indicates that there is a 5% probability that some neuron connections are randomly disabled during training. This is especially important when the data volume is limited.
- **target_modules**: specifies the weight matrices to which LoRA applies in the model by using a regular expression. In Llama, the configuration here applies LoRA to the Query (WQ), Key (WK), Value (WV), and Output (WO) matrices in the self-attention mechanism of the model. These matrices play a key role in the Transformer structure. After LoRA is inserted, the model performance can be maintained while the number of parameters is reduced.

By configuring these parameters, LoRA can effectively reduce the computing resource usage during fine-tuning while maintaining the high performance of the model.

### Examples of LoRA Fine-Tuning for Llama2-7B

MindFormers provides [the LoRA fine-tuning examples](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#lora%E5%BE%AE%E8%B0%83) of Llama2-7B. For details about the dataset used during fine-tuning, see [dataset downloading](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

Take Llama2-7B as an example. You can run the following **msrun** startup script to perform 8-device distributed fine-tuning.

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt False \
 --use_parallel True \
 --run_mode finetune" 8
```

If the weight can be loaded only after conversion, the weight loading path must be set to the upper-layer path of **rank_0** and the automatic weight conversion function must be enabled (**--auto_trans_ckpt** is set to **True**).

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/checkpoint/ \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```

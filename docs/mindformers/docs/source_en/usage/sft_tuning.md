# Supervised Fine-Tuning (SFT)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/usage/sft_tuning.md)

## Overview

SFT (Supervised Fine-Tuning) employs supervised learning ideas and refers to the process of adjusting some or all of the parameters based on a pre-trained model to make it more adaptable to a specific task or dataset.

## Process

SFT consists of the following steps:

- **Pretraining:**
  A neural network model is trained on a large-scale dataset. For example, an LLM is trained on a large amount of unlabeled text data. The objective of the pre-training phase is to enable the model to obtain common knowledge and understanding capabilities.
- **Fine-tuning:**
  Based on the target task, the obtained pretrained model is fine-tuned by using the new training dataset. During fine-tuning, all or some parameters of the original model can be optimized through backpropagation to achieve a better effect of the model on the target task.
- **Evaluation:**
  After fine-tuning, a new model is obtained. The fine-tuning model may be evaluated by using the evaluation dataset of the target task to obtain performance metrics of the fine-tuning model on the target task.

Based on actual operations, SFT may be decomposed into the following steps:

1. **Selecting a pretrained model:**
   Select a pretrained language model, for example, GPT-2 or Llama2. The pretrained model is trained on a large text corpus to learn a general representation of a language.
2. **Downloading the model weights:**
   For the selected pretrained model, download the pretrained weights from the HuggingFace model library.
3. **Converting model weights:**
   Convert the downloaded HuggingFace weight based on the required framework, for example, convert it to the CKPT weights supported by the MindSpore framework.
4. **Preparing a dataset:**
   Select a dataset for fine-tuning tasks based on the fine-tuning objective. For LLMs, the fine-tuning dataset is data that contains text and labels, for example, the alpaca dataset. When using a dataset, you need to preprocess the corresponding data. For example, when using the MindSpore framework, you need to convert the dataset to the MindRecord format.
5. **Performing a fine-tuning task:**
   Use the dataset of the fine-tuning task to train the pre-trained model and update the model parameters. If all parameters are fine-tuned, all parameters are updated. After the fine-tuning task is complete, a new model can be obtained.

## SFT Fine-Tuning Methods

MindSpore Transformers currently supports two SFT fine-tuning methods: full-parameter fine-tuning and LoRA low-parameter fine-tuning. Full-parameter fine-tuning refers to updating all parameters during training, which is suitable for large-scale data fine-tuning, and can get the optimal adaptability to the task, but requires larger computational resources.LoRA low-parameter fine-tuning only updates some parameters during training, which uses less memory and is faster than full-parameter fine-tuning, but is not as effective as full-parameter fine-tuning in some tasks.

### Introduction to the LoRA Principle

LoRA achieves a significant reduction in the number of parameters by decomposing the weight matrix of the original model into two low-rank matrices. For example, suppose a weight matrix W has size m x n. With LoRA, this matrix is decomposed into two low-rank matrices A and B, where A has size m x r and B has size r x n (r is much smaller than m and n). During the fine-tuning process, only these two low-rank matrices are updated without changing the rest of the original model.

This approach not only drastically reduces the computational overhead of fine-tuning, but also preserves the original performance of the model, which is especially suitable for model optimization in environments with limited data volume and restricted computational resources. For detailed principles, you can check the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

## Using MindSpore Transformers for Full-Parameter Fine-Tuning

### Selecting a Pretrained Model

MindSpore Transformers supports mainstream foundation models in the industry. This practice uses the Llama2-7B model for SFT as an example.

### Downloading the Model Weights

MindSpore Transformers provides pretrained weights and vocabulary files that have been converted for pretraining, fine-tuning, and inference. You can also download the official HuggingFace weights and convert model weights before using these weights.

You can download the vocabulary at [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model).

| Model     |    MindSpore Weight   |     HuggingFace Weight        |
|:----------|:------------------------:| :----------------------: |
| Llama2-7B |  [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)      | [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

> All weights of Llama2 need to be obtained by [submitting an application](https://ai.meta.com/resources/models-and-libraries/llama-downloads) to Meta. If necessary, apply for the weights by yourself.

### Converting Model Weights

Take the [Llama2-7B model](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) as an example. The original HuggingFace weight file contains the following information:<br>

- `config.json`: main configuration information of the model architecture.<br>
- `generation_config.json`: configuration information related to text generation.<br>
- `safetensors file`: model weight file.<br>
- `model.safetensors.index.json`: JSON file that describes safetensors model parameter file index and model slices.<br>
- `bin file`: PyTorch model weight file.<br>
- `pytorch_model.bin.index.json`: JSON file that describes PyTorch index and model slices.<br>
- `tokenizer.json`: tokenizer vocabulary configuration file.<br>
- `tokenizer.model`: tokenizer of the model.<br>

MindSpore Transformers provides a weight conversion script. You can run the conversion script [convert_weight.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/convert_weight.py) to convert the HuggingFace weights to the complete CKPT weights.

```bash
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME
```

Parameters:

```commandline
model:       model name. For details about other models, see the model description document.
input_path:  path of the folder where the HuggingFace weight is downloaded.
output_path: path for storing the converted MindSpore weight file.
```

### Preparing a Dataset

MindSpore Transformers provides **WikiText2** as the pretraining dataset and **alpaca** as the fine-tuning dataset.

| Dataset    |                 Applicable Model                 |   Applicable Phase   |              Download Link     |
|:----------|:-------------------------------------:|:---------:| :--------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| alpaca    | Llama2-7B<br>Llama2-13B<br>Llama2-70B |    Fine-tuning    |                   [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                   |

The following uses the alpaca dataset as an example. After downloading the dataset, you need to preprocess it. For details about how to download the `tokenizer.model` used in preprocessing, see the model weight download.

**alpaca Data Preprocessing**

1. Run the [alpaca_converter.py script](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py) in MindSpore Transformers to convert the dataset into the multi-round dialog format.

    ```bash
    python alpaca_converter.py \
      --data_path /{path}/alpaca_data.json \
      --output_path /{path}/alpaca-data-conversation.json
    ```

    Parameters:

    ```commandline
    data_path:   path of the file to be downloaded.
    output_path: path for storing output files.
    ```

2. Run the [llama_preprocess.py script](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py) in MindSpore Transformers to convert the data into the MindRecord format. This operation depends on the fastchat tool package to parse the prompt template. You need to install fastchat 0.2.13 or later in advance.

    ```bash
    python llama_preprocess.py \
      --dataset_type qa \
      --input_glob /{path}/alpaca-data-conversation.json \
      --model_file /{path}/tokenizer.model \
      --seq_length 4096 \
      --output_file /{path}/alpaca-fastchat4096.mindrecord
    ```

    Parameters:

    ```commandline
    dataset_type: type of the data to be preprocessed.
    input_glob:   path of the converted alpaca file.
    model_file:   path of the tokenizer.model file.
    seq_length:   sequence length of the output data.
    output_file:  path for storing output files.
    ```

### Performing a Fine-tuning Task

#### Single-Card Training

Execute `run_mindformer.py` to start the fine-tuning task on a single card. Below is an example usage:

Taking the fine-tuning of the Llama2 model on a single card as an example, due to the limited NPU memory, it is not possible to run the full Llama2-7B model, so we reduce the layers for the example. Modify `finetune_llama2_7b.yaml` and set `num_layers` to 2.

The startup command is as follows:

```shell
python run_mindformer.py \
 --config configs/llama2/finetune_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --use_parallel False \
 --run_mode finetune
```

#### Single-Node Training

Take Llama2-7B as an example. Run the startup script **msrun** to perform 8-device distributed training. The startup command is as follows:

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/finetune_llama2_7b.yaml \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --use_parallel True \
 --run_mode finetune" 8
```

Parameters:

```commandline
config:            model configuration file, which is stored in the config directory of the MindSpore Transformers code repository.
load_checkpoint:   path of the checkpoint file.
train_dataset_dir: path of the training dataset.
use_parallel:      specifies whether to enable parallelism.
run_mode:          running mode. The value can be train, finetune, or predict (inference).
```

After the task is executed, the **checkpoint** folder is generated in the **mindformers/output** directory, and the model file is saved in this folder.

#### Multi-Node Training

The multi-node multi-device fine-tuning task is similar to the pretrained task. You can refer to the [multi-node multi-device pretraining command](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/usage/pre_training.html#multi-node-training) and modify the command as follows:

1. Add the input parameter `--load_checkpoint /{path}/llama2_7b.ckpt` to the startup script to load the pretrained weights.
2. Set `--train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord` in the startup script to load the fine-tuning dataset.
3. Set `--run_mode finetune` in the startup script. **run_mode** indicates the running mode, whose value can be **train**, **finetune**, or **predict** (inference).

After the task is executed, the **checkpoint** folder is generated in the **mindformers/output** directory, and the model file is saved in this folder.

## Using MindSpore Transformers for LoRA Low-Parameter Fine-Tuning

MindSpore Transformers supports configurable enablement of LoRA fine-tuning, which eliminates the need for code adaptation for each model and can be used to perform LoRA low-parameter fine-tuning tasks by simply modifying the model configuration in the YAML configuration file for full-parameter fine-tuning and adding the `pet_config` low-parameter fine-tuning configuration. The following shows the model configuration section of the YAML configuration file for LoRA fine-tuning of the Llama2 model, with a detailed description of the `pet_config` parameter.

### YAML File Example

For details about the complete YAML file, see [the Llama2 LoRA fine-tuning YAML file](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/lora_llama2_7b.yaml).

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

### Examples of LoRA Fine-Tuning for Llama2-7B

MindSpore Transformers provides [the LoRA fine-tuning examples](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#lora%E5%BE%AE%E8%B0%83) of Llama2-7B. For details about the dataset used during fine-tuning, see [dataset downloading](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

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

When the distributed strategy of the weights does not match the distributed strategy of the model, the weights need to be transformed. The load weight path should be set to the upper path of the directory named with `rank_0`, and the weight auto transformation function should be enabled by setting `--auto_trans_ckpt True` . For a more detailed description of the scenarios and usage of distributed weight transformation, please refer to [Distributed Weight Slicing and Merging](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/function/transform_weight.html).

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/checkpoint/ \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```

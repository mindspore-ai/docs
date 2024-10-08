# Supervised Fine-Tuning (SFT)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/usage/sft_tuning.md)

## Overview

SFT uses supervised learning. Pretraining is performed with a source dataset to obtain an original model, and then parameters of the original model are fine-tuned with a new dataset to obtain a new model, achieving better performance on new tasks.

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

## MindFormers-based Full-Parameter Fine-Tuning Practice

### Selecting a Pretrained Model

MindFormers supports mainstream foundation models in the industry. This practice uses the Llama2-7B model for SFT as an example.

### Downloading the Model Weights

MindFormers provides pretrained weights and vocabulary files that have been converted for pretraining, fine-tuning, and inference. You can also download the official HuggingFace weights and convert model weights before using these weights.

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

MindFormers provides a weight conversion script. You can run the conversion script [convert_weight.py](https://gitee.com/mindspore/mindformers/blob/dev/convert_weight.py) to convert the HuggingFace weights to the complete CKPT weights.

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

MindFormers provides **WikiText2** as the pretraining dataset and **alpaca** as the fine-tuning dataset.

| Dataset    |                 Applicable Model                 |   Applicable Phase   |              Download Link     |
|:----------|:-------------------------------------:|:---------:| :--------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| alpaca    | Llama2-7B<br>Llama2-13B<br>Llama2-70B |    Fine-tuning    |                   [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                   |

The following uses the alpaca dataset as an example. After downloading the dataset, you need to preprocess it. For details about how to download the `tokenizer.model` used in preprocessing, see the model weight download.

**alpaca Data Preprocessing**

1. Run the [alpaca_converter.py script](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py) in MindFormers to convert the dataset into the multi-round dialog format.

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

2. Run the [llama_preprocess.py script](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py) in MindFormers to convert the data into the MindRecord format. This operation depends on the fastchat tool package to parse the prompt template. You need to install fastchat 0.2.13 or later and Python 3.9 in advance.

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
config:            model configuration file, which is stored in the config directory of the MindFormers code repository.
load_checkpoint:   path of the checkpoint file.
train_dataset_dir: path of the training dataset.
use_parallel:      specifies whether to enable parallelism.
run_mode:          running mode. The value can be train, finetune, or predict (inference).
```

After the task is executed, the **checkpoint** folder is generated in the **mindformers/output** directory, and the model file is saved in this folder.

#### Multi-Node Training

The multi-node multi-device fine-tuning task is similar to the pretrained task. You can refer to the multi-node multi-device pretraining command and modify the command as follows:

1. Add the input parameter `--load_checkpoint /{path}/llama2_7b.ckpt` to the startup script to load the pretrained weights.
2. Set `--train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord` in the startup script to load the fine-tuning dataset.
3. Set `--run_mode finetune` in the startup script. **run_mode** indicates the running mode, whose value can be **train**, **finetune**, or **predict** (inference).

After the task is executed, the **checkpoint** folder is generated in the **mindformers/output** directory, and the model file is saved in this folder.

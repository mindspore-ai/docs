# Ckpt Weights

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/ckpt.md)

## Overview

Ckpt is a common file format used to save model training status in the deep learning framework. It contains model parameters, optimizer status, and training progress. It is used to restore training or fine-tune models. This document describes how MindSpore Transformers supports conversion , slice and merge.

> The ckpt format is planned to offline. The safetensors format is recommended for weights. Safetensors is a reliable and portable machine learning model storage format from Huggingface for storing Tensors securely and with fast storage (zero copies). For details, see [Safetensors Weights](https://www.mindspore.cn/mindformers/docs/en/dev/feature/safetensors.html).

## Weight Format Conversion

### Overview

MindSpore Transformers provides a unified weight conversion tool that allows model weights to convert between the HuggingFace and MindSpore Transformers formats. This helps you:

- Convert a HuggingFace weight to a MindSpore Transformers one for fine-tuning, evaluation, or inference on MindSpore Transformers.
- Convert the weights trained or fine-tuned using MindSpore Transformers to HuggingFace weights and uses them on other frameworks.

### Conversion Procedure

To perform weight conversion, clone the complete HuggingFace repository of the model to be converted locally, and execute the `mindformers/convert_weight.py` script. This script automatically converts the HuggingFace model weight file into a weight file applicable to MindSpore Transformers. If you want to convert a MindSpore Transformers weight to a HuggingFace one, set `reversed` to `True`.

```shell
python convert_weight.py [-h] --model MODEL [--reversed] --input_path INPUT_PATH  --output_path OUTPUT_PATH [--dtype DTYPE] [--n_head N_HEAD] [--hidden_size HIDDEN_SIZE] [--layers LAYERS] [--is_pretrain IS_PRETRAIN] [--telechat_type TELECHAT_TYPE]
```

#### Parameters

- model: model name.
- reversed: converts a MindSpore Transformers weight to the HuggingFace one.
- input_path: path of the HuggingFace weight folder, which points to the downloaded weight file.
- output_path: path for storing the MindSpore Transformers weight file after conversion.
- dtype: weight data type after conversion.
- n_head: takes effect only for the BLOOM model. Set this parameter to `16` when `bloom_560m` is used and to `32` when `bloom_7.1b` is used.
- hidden_size: takes effect only for the BLOOM model. Set this parameter to `1024` when `bloom_560m` is used and to `4096` when `bloom_7.1b` is used.
- layers: number of layers to be converted. This parameter takes effect only for the GPT2 and WizardCoder models.
- is_pretrain: converts the pre-trained weight. This parameter takes effect only for the Swin model.
- telechat_type: version of the TeleChat model. This parameter takes effect only for the TeleChat model.

### Conversion Example

Assume that you have downloaded the [Llama2 model weight](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD) and saved it in the `/home/user/torch_weights` path, to convert it to the MindSpore Transformers weight and save it in the `/home/user/ms_weights` path, run the following command:

```bash
python convert_weight.py --model llama --input_path /home/user/torch_weights --output_path /home/user/ms_weights/llama.ckpt
```

After the preceding steps are performed, the HuggingFace weight is successfully converted to a MindSpore Transformers weight, facilitating model training or inference on MindSpore Transformers.

### Supported Models

| Parameter Value      | Supported models                            |
|-----------|---------------------------------------------|
| llama     | Llama2, Llama3, Llama3.1, CodeLlama         |
| baichuan2 | Baichuan2                                   |
| glm-n     | GLM2, GLM3, GLM3-32K, GLM4                  |
| cogvlm2   | CogVLM2-Video, CogVLM2-Image                |
| qwen      | Qwen, Qwen1.5, Qwen2                        |
| qwenvl    | QwenVL                                      |
| internlm  | InternLM                                    |
| internlm2 | InternLM2                                   |
| yi        | Yi                                          |
| mixtral   | Mixtral                                     |
| deepseek  | DeepSeekCoder, DeepSeekCoder1.5, DeepSeekV2 |
| gpt       | GPT2                                        |
| whisper   | Whisper                                     |

### Developing Weight Conversion for Unsupported Models

1. Add the `convert_weight.py` and `convert_reversed.py` files to the extended model directory.
2. Compile the `convert_pt_to_ms` and `convert_ms_to_pt` weight conversion functions in the files. The function parameters are `input_path`, `output_path`, `dtype`, and an additional parameter `**kwargs`.
3. Add the extended model name and conversion function import paths to the `convert_map` and `reversed_convert_map` dictionaries in the `convert_weight.py` file in the MindSpore Transformers code root directory.
4. Call the `parser.add_argument()` method in the `main` function to add the additional parameter.

### Example of Developing Model Weight Conversion

Llama is used as an example. To convert a HuggingFace weight to a MindSpore Transformers one, define the `convert_pt_to_ms` function in [convert_weight.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_weight.py).

```python
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import LlamaForCausalLM
    except:
        raise ImportError(f"Failed to load huggingface checkpoint. Please make sure transformers is available.")

    try:
        model_hf = LlamaForCausalLM.from_pretrained(os.path.dirname(input_path))
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name[:7] == 'layers.':
            name = name[7:]

        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
```

To convert a MindSpore Transformers weight to a HuggingFace one, define the `convert_ms_to_pt` function in [convert_reversed.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_reversed.py).

```python
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert ms weight to hf."""
    print(f"Trying to convert mindspore checkpoint in '{input_path}'.", flush=True)
    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        name = name_replace(name)
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        if is_lora_param(name):
            name = name.replace('.tk_delta_lora_a', '.lora_A.weight')
            name = name.replace('.tk_delta_lora_b', 'lora_B.weight')
        state_dict[name] = ms2pt(value, dtype)

    torch.save(state_dict, output_path)
    print(f"\rConvert mindspore checkpoint finished, the huggingface checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
```

## Distributed Weight Slicing and Merging

### Overview

In a current distributed training and inference environment, if a pre-trained weight does not match a distributed strategy, the pre-trained weight needs to be converted to adapt to the corresponding distributed strategy. MindSpore Transformers provides a set of weight conversion tools to meet the requirements in different scenarios. This tool can be used to slice a single-device weight into multi-device weights, convert between multi-device weights, and merge multi-device weights into a single-device weight. You can select [Automatic Conversion](#automatic-conversion) or [Offline Conversion](#offline-conversion) as required so that a model can quickly switch between different distributed scenarios.

In addition, MindSpore Transformers supports [LoRA Weight Merging](#lora-weight-merging) to facilitate the deployment of models fine-tuned using LoRA.

### Automatic Conversion

When a model loads a weight, it automatically checks whether the weight is matching the distributed slicing strategy of the current model. If they do not match, the weight is automatically converted.

#### Parameters

Parameters in the `yaml` file related to **automatic weight conversion** are described as follows:

| Parameter             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| load_checkpoint     | Absolute path or folder path of the pre-loaded weights.<br> - For a complete set of weights, set this parameter to an absolute path.<br> - For a distributed weight, set this parameter to the folder path. The distributed weight must be stored in the `model_dir/rank_x/xxx.ckpt` format. The folder path is `model_dir`.<br>**If there are multiple CKPT files in the rank_x folder, the last CKPT file in the file name sequence is used for conversion by default.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| src_strategy_path_or_dir        | Path of [the distributed strategy file](#offline-conversion-configuration) corresponding to the pre-loaded weights.<br> - If the pre-loaded weights are a complete set of weights, leave this parameter **blank**.<br> - If the pre-loaded weights are distributed and pipeline parallelism is used when the pre-loaded weights are saved, set this parameter to the **merged strategy file path** or **distributed strategy folder path**.<br> - If the pre-loaded weights are distributed and pipeline parallelism is not used when the pre-load weights are saved, set this parameter to any **ckpt_strategy_rank_x.ckpt** path.                                                                                                                                                                                                                                                                                                                                                                                             |
| auto_trans_ckpt     | Specifies whether to enable automatic weight conversion. The value True indicates that it is enabled. The default value is False.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| transform_process_num | Number of processes used for automatic weight conversion. The default value is 1.<br> - If transform_process_num is set to 1, only rank_0 is used for weight conversion. Other processes wait until the conversion ends.<br> - If transform_process_num is larger than 1, **multiple processes conduct conversion**. For example, for an 8-device task, if transform_process_num is set to 2, rank_0 is used for converting the weights of slices rank_0, rank_1, rank_2, and rank_3, and rank_4 is used for converting the weights of slices rank_4, rank_5, rank_6, and rank_7, and other processes wait until rank_0 and rank_4 complete the conversion.<br>**Note**:<br> 1. A larger value of transform_process_num indicates a shorter conversion time and **a larger host memory occupied by the conversion**. If the host memory is insufficient, decrease the value of transform_process_num.<br> 2. The value of transform_process_num must be a number that can be exactly divided by and cannot exceed that of NPUs. |
| transform_by_rank   | Specifies whether to use the mindspore.transform_checkpoint_by_rank API for weight conversion.<br> - If transform_process_num is larger than 1, the value is automatically set to `True`.<br> - If transform_process_num is set to 1, if the target weight is a distributed weight, the mindspore.transform_checkpoint_by_rank API is cyclically called to convert the weight of each rank slice in serial mode.<br>- If transform_process_num is set to 1, if the target weight is a complete weight, the value is automatically set to `False`, and the mindspore.transform_checkpoints API is called for weight conversion.                                                                                                                                                                                                                                                                                                                                                                                                  |

#### YAML Configurations in Different Scenarios

**Slicing a Single-Device Weight into Multi-Device Weights**

```yaml
# load_checkpoint: specifies path of the pre-trained weight file.
load_checkpoint: "/worker/llama3_8b/llama3_8b.ckpt"

# auto_trans_ckpt: specifies whether to enable automatic conversion.
auto_trans_ckpt: True
```

**Conversion Between Multi-Device Weights**

```yaml
# load_checkpoint: specifies the path of the multi-device weight folder.
load_checkpoint: "/worker/checkpoint/llama3-8b-2layer-dp2mp2pp2"

# src_strategy_path_or_dir: specifies the path of the distributed strategy file.
src_strategy_path_or_dir: "/worker/checkpoint/llama3-8b-2layer-dp2mp2pp2/strategy/merged_ckpt_strategy.ckpt"

# auto_trans_ckpt: specifies whether to enable automatic conversion.
auto_trans_ckpt: True
```

**Merging Multi-Device Weights into a Single-Device Weight**

```yaml
# load_checkpoint: specifies the path of the multi-device weight folder.
load_checkpoint: "/worker/checkpoint/llama3-8b-2layer-dp1mp2pp2"

# src_strategy_path_or_dir: specifies the path of the distributed strategy file.
src_strategy_path_or_dir: "/worker/checkpoint/llama3-8b-2layer-dp1mp2pp2/strategy/merged_ckpt_strategy.ckpt"

# auto_trans_ckpt: specifies whether to enable automatic conversion.
auto_trans_ckpt: True

# use_parallel: Set it to False.
use_parallel: False
```

**Enabling Multi-Process Conversion (Optional)**

```yaml
# transform_process_num: specifies the number of processes involved in the conversion.
transform_process_num: 2
```

#### Precautions

- **Multi-process conversion**: Set the `transform_process_num` parameter to enable multi-process conversion. Pay attention to the memory usage. If a memory overflow occurs, you are advised to reduce the number of processes.

- **Automatic weight conversion**: After this function is enabled, the system deletes the old `strategy` and `transformed_checkpoint` folders from the `output` directory and saves the output of the current task. After the conversion task is complete, you are advised to move the `strategy` and `transformed_checkpoint` folders to a user-defined directory to prevent them from being deleted by mistake in subsequent operations.

- **Distributed strategy file saving**: The distributed strategy file is saved in the `output/strategy` folder. If **pipeline parallelism** is enabled, the system automatically merges all `ckpt_strategy_rank_x.ckpt` files into a `merged_ckpt_strategy.ckpt` file. If pipeline parallelism is not enabled, the MERGE operation is not performed.

### Offline Conversion

The offline conversion function is designed to meet your requirements for manually converting weights. With offline conversion, you can convert model weights in an independent environment. Offline conversion supports multiple weight conversion scenarios, including slicing a single-device weight into multi-device weights, converting between multi-device weights, and merging multi-device weights into a single-device weight.

When using offline conversion, you can manually configure conversion parameters as required to ensure that the conversion process is flexible and controllable. This function is especially suitable for model deployment and optimization in a strictly controlled computing environment.

#### Parameters

Parameters in the `yaml` file related to **offline weight conversion** are described as follows:

| Parameter       | Description                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| src_checkpoint | Absolute path or folder path of the source weight.<br> - For **a complete set of weights**, set this parameter to an **absolute path**.<br> - For **distributed weights**, set this parameter to the **folder path**. The distributed weights must be stored in the `model_dir/rank_x/xxx.ckpt` format. The folder path is `model_dir`.<br>**If there are multiple CKPT files in the rank_x folder, the last CKPT file in the file name sequence is used for conversion by default.**             |
| src_strategy_path_or_dir   | Path of the distributed strategy file corresponding to the source weight.<br> - For a complete set of weights, leave it **blank**.<br> - For distributed weights, if pipeline parallelism is used, set this parameter to the **merged strategy file path** or **distributed strategy folder path**.<br> - For distributed weights, if pipeline parallelism is not used, set this parameter to any **ckpt_strategy_rank_x.ckpt** path.        |
| dst_checkpoint | Path of the folder that stores the target weight.  |
| dst_strategy   | Path of the distributed strategy file corresponding to the target weight.<br> - For a complete set of weights, leave it **blank**.<br> - For distributed weights, if pipeline parallelism is used, set this parameter to the **merged strategy file path** or **distributed strategy folder path**.<br> - For distributed weights, if pipeline parallelism is not used, set this parameter to any **ckpt_strategy_rank_x.ckpt** path.|
| prefix          | Prefix name of the saved target weight. The weight is saved as {prefix}rank_x.ckpt. The default value is checkpoint_.    |
| world_size     | Total number of slices of the target weight. Generally, the value is dp \* mp \* pp.    |
| process_num    | Number of processes used for offline weight conversion. The default value is 1.<br> - If process_num is set to 1, **a single process is used for conversion**.<br>- If process_num is larger than 1, **multi-process conversion** is used. For example, if the target weight for conversion is the distributed weight of eight GPUs and process_num is set to 2, two processes are started to convert the weights of slices rank_0, rank_1, rank_2, and rank_3 and slices rank_4, rank_5, rank_6, and rank_7, respectively.  |

#### Offline Conversion Configuration

**Generating Distributed Strategy**

MindSpore generates a distributed strategy file (ckpt format) corresponding to the number of cards in the `output/strategy` folder after running a distributed task, which can be used in offline weight conversion.

If there is currently no distributed strategy file, it can be quickly generated by setting `only_save_strategy:True` in the yaml configuration file on the basis of the original distributed training/inference task. After setting, the task will stop immediately after generating the distributed strategy file, without actually executing training or inference.

**Single-Process Conversion**

Use [mindformers/tools/ckpt_transform/transform_checkpoint.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.py) to perform single-process conversion on the loaded weight.

**Run the command.**

```shell
python transform_checkpoint.py \
  --src_checkpoint /worker/checkpoint/llama3-8b-2layer/rank_0/llama3_8b.ckpt \
  --dst_checkpoint /worker/transform_ckpt/llama3_8b_1to8/ \
  --dst_strategy /worker/mindformers/output/strategy/
```

**Multi-Process Conversion**

Use [mindformers/tools/ckpt_transform/transform_checkpoint.sh](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.sh) to perform multi-process conversion on the loaded weight.

**Run the command.**

```shell
bash transform_checkpoint.sh \
  /worker/checkpoint/llam3-8b-2layer/rank_0/llama3_8b.ckpt \
  None \
  /worker/transform_ckpt/llama3_8b_1to8/ \
  /worker/mindformers/output/strategy/ \
  8 2
```

**Precautions**:

- When the [transform_checkpoint.sh](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.sh) script is used, `8` indicates the number of target devices, and `2` indicates that two processes are used for conversion.

### Special Scenarios

#### Multi-Node Multi-Device Training on Physical Machines

Training a large-scale model usually needs a cluster of servers. In the multi-node multi-device scenario, if there is a shared disk between servers, the automatic conversion function can be used. Otherwise, only offline conversion can be used. The following example is a training that uses two servers and 16 GPUs.

**Scenario 1: A shared disk exists between servers.**

If there is a shared disk between servers, you can use MindSpore Transformers to automatically convert a weight before multi-node multi-device training. Assume that `/data` is the shared disk between the servers and the MindSpore Transformers project code is stored in the `/data/mindformers` directory.

- **Single-process conversion**

  In single-process conversion mode, you only need to set the path of the pre-trained weight in the configuration file and enable automatic weight conversion.

  **Configure the parameter.**

  ```yaml
  # Set the path of the pre-trained weight file to an absolute path.
  load_checkpoint: "/worker/checkpoint/llama3-8b/rank_0/llama3_8b.ckpt"

  # Set auto_trans_ckpt to True to enable automatic weight conversion.
  auto_trans_ckpt: True

  # Set the dataset path.
  train_dataset: &train_dataset
    data_loader:
      type: MindDataset
      dataset_dir: "/worker/dataset/wiki103/"
      shuffle: True

  # Configure the 16-device distributed strategy (for reference only).
  parallel_config:
    data_parallel: 2
    model_parallel: 4
    pipeline_stage: 2
    micro_batch_num: 2
    vocab_emb_dp: True
    gradient_aggregation_group: 4
    micro_batch_interleave_num: 1
  ```

- **Multi-process conversion (optional)**

  To accelerate weight conversion, you can choose the multi-process conversion mode by setting the `transform_process_num` parameter.

  **Configure the parameter.**

  ```yaml
  # Use two processes for conversion.
  transform_process_num: 2
  ```

  **Start a task.**

  Use [mindformers/scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh) to start the task.

  ```shell
  # First server (main node)
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 0 output/msrun_log False 300
  # Second server (subnode)
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 1 output/msrun_log False 300
  ```

**Scenario 2: No shared disk exists between servers.**

If there is no shared disk between servers, you need to use the offline weight conversion tool to convert the weight. The following steps describe how to perform offline weight conversion and start a multi-node multi-device training task.

- **Obtain the distributed policy file.**

  Before offline weight conversion, you need to obtain the distributed strategy file of each node.

  **Configure the parameter.**

  ```yaml
  # Set **only_save_strategy** to **True** to obtain the distributed strategy file.
  only_save_strategy: True

  # Set the dataset path.
  train_dataset: &train_dataset
    data_loader:
      type: MindDataset
      dataset_dir: "/worker/dataset/wikitext_2048/"
      shuffle: True

  # Configure the 16-device distributed strategy (for reference only).
  parallel_config:
    data_parallel: 2
    model_parallel: 4
    pipeline_stage: 2
    micro_batch_num: 2
    vocab_emb_dp: True
    gradient_aggregation_group: 4
    micro_batch_interleave_num: 1
  ```

  The strategy file of each node is stored in the corresponding `output/strategy` directory. For example, node 0 stores the `ckpt_strategy_rank_0-7.ckpt` file, and node 1 stores the `ckpt_strategy_rank_8-15.ckpt` file. Then, you need to integrate the strategy files of all nodes on the same server to facilitate subsequent operations.

- **Offline weight conversion**

  On the server where all strategy files are stored, use [mindformers/tools/ckpt_transform/transform_checkpoint.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.py) to perform offline weight conversion.

  **Single-process conversion**

  ```shell
  python mindformers/tools/ckpt_transform/transform_checkpoint.py \
    --src_checkpoint /worker/checkpoint/llama3-8b/rank_0/llama_7b.ckpt \
    --dst_checkpoint ./output/llama3_8b_dp2mp4pp2 \
    --dst_strategy ./output/strategy
  ```

  **Multi-process conversion (optional)**

  ```shell
  # Use two processes for conversion.
  bash mindformers/tools/ckpt_transform/transform_checkpoint.sh \
    /worker/checkpoint/llama3-8b/rank_0/llama_7b.ckpt \
    None \
    ./output/llama3_8b_dp2mp4pp2 \
    ./output/strategy \
    16 2
  ```

- **Copy the weights to other nodes.**

  Copy the distributed weights that have been converted to respective nodes. Node 0 requires only the weights of slices from `rank_0` to `rank_7`, and node 1 requires only the weights of slices from `rank_8` to `rank_15`.

- **Set the parameter.**

  ```yaml
  # Set the pre-trained weight path to model_dir, the distributed weight folder path.
  load_checkpoint: "/worker/checkpoint/llama3_8b_dp2mp4pp2"

  # Change only_save_strategy to False.
  only_save_strategy: False
  ```

#### ModelArts Training

Training in ModelArts is similar to multi-node multi-device training on physical machines. Automatic weight conversion can also be enabled. You can set `auto_trans_ckpt=True` in the hyperparameters of a training task to enable automatic weight conversion and set `transform_process_num > 1` to enable multi-process conversion.

**Note**: If the number of NPUs on the server node in the ModelArts resource pool is not 8, you need to set `npu_num_per_node = the number of NPUs on the node`. For example, if each node is configured with 16 NPUs, `npu_num_per_node=16` should be set.

### LoRA Weight Merging

#### Overview

The basic principle of low-rank adaptation (LoRA) is to parameterize the original model with low-rank weights. The core process of merging LoRA weights is to calculate the parameters of the LoRA branches and add them to the corresponding model parameters, which makes the parameter list of the final weight file the same as that of the original model and excludes additional LoRA parameters. This operation does not affect the inference result. Therefore, the model after merging still has the same performance as the original model during inference.
For details about the principles and implementation of LoRA, see the following resources:

- Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- GitHub: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

#### Instructions

Use the [LoRA weight merging script](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/transform_ckpt_lora.py) provided by MindSpore Transformers to merge LoRA weights as follows:

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_strategy src_strategy_path_or_dir \
  --src_ckpt_path_or_dir src_ckpt_path_or_dir \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

**Parameters**

- **src_ckpt_strategy**: specifies the path of the distributed strategy file corresponding to the source weight. The file is stored in the `output/strategy/` directory by default after the training task is started. If the source is a complete set of weights, you do not need to set this parameter. If the source contains distributed weights, set this parameter based on the following conditions:
    - **Pipeline parallelism enabled for the source weights**: Weight conversion is based on the merging strategy file. Set the parameter to the path of the distributed strategy folder. The script automatically merges all `ckpt_strategy_rank_x.ckpt` files in the folder into `merged_ckpt_strategy.ckpt` in the folder. If `merged_ckpt_strategy.ckpt` already exists, set the parameter to the path of the file.
    - **Pipeline parallelism not enabled for the source weights**: Weight conversion can be based on any strategy file. Set the parameter to the path of any `ckpt_strategy_rank_x.ckpt` file.

    **Note**: If a `merged_ckpt_strategy.ckpt` already exists in the strategy folder and is still transferred to the folder path, the script deletes the old `merged_ckpt_strategy.ckpt` and then merges files into a new `merged_ckpt_strategy.ckpt` for weight conversion. Therefore, ensure that the folder has enough write permission. Otherwise, an error will be reported.
- **src_ckpt_path_or_dir**: specifies the path of the source weight. For distributed weights, set the parameter to the path of the folder where the source weights are located. The source weights must be stored in the `model_dir/rank_x/xxx.ckpt` format, and the folder path must be set to `model_dir`. If the source is a complete set of weights, set the parameter to an absolute path.
- **dst_ckpt_dir**: specifies the path for storing the target weight, which must be a user-defined path of an empty folder. The target weight is saved in the `model_dir/rank_x/xxx.ckpt` format.
- **prefix**: name prefix of the target weight file. The default value is "checkpoint_", indicating that the target weight is saved in the `model_dir/rank_x/checkpoint_x.ckpt` format.
- **lora_scaling**: combination coefficient of the LoRA weight. The default value is `lora_alpha/lora_rank`. The two parameters are used for LoRA model configuration and need to be calculated.

#### Examples

**Scenario 1: There is a complete set of weights for LoRA parameters.**

If the weight file before merging is a complete one, you can set the parameters as follows (directly enter the path of the complete set of weights):

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_path_or_dir .../xxx/xxx.ckpt \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

**Scenario 2: There are distributed weights for LoRA parameters.**

If the weight file before merging contains distributed weights, you can set the parameters as follows (enter the path of the distributed weight folder and the path of the distributed strategy folder). The obtained weights are automatically merged into a complete weight file.

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_strategy .../xxx/mindformers/output/strategy/ \
  --src_ckpt_path_or_dir .../xxx/model_dir \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

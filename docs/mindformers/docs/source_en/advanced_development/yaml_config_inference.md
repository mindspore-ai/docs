# Guide to Using the Inference Configuration Template

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/advanced_development/yaml_config_inference.md)

## Overview

Currently, the Mcore architecture model supports reading the Hugging Face model directory for model instantiation during inference. Therefore, MindSpore Transformers has streamlined the model's YAML configuration files. Instead of having a separate YAML file for each model and each specification, they have been unified into a single YAML configuration template. For online inference of models with different specifications, you only need to apply the configuration template, set up the model directory downloaded from Hugging Face or ModelScope, and modify a few necessary configurations to perform inference.

## Usage Method

When using the inference configuration template for inference, some configurations in it need to be modified according to the actual situation.

### Configurations that Must be Modified (Required)

The configuration template does not contain model configurations; it relies on reading model configurations from Hugging Face or ModelScope to instantiate the model. Therefore, the following configurations must be modified:

| Configuration Item | Configuration Description | Modification Method |
|----|----|--------|
|pretrained_model_dir|Path to the model directory|Change it to the folder path of the model file downloaded from Hugging Face or ModelScope.|

### Optional Scenario-Based Configuration (Optional)

The following different usage scenarios require modifications to some configurations:

#### Default Scenario (single card, 64GB video memory)

The inference configuration template is by default set for scenarios with a single card and 64GB of video memory, and no additional configuration modifications are needed in this case. It should be noted that if the model scale is too large and the single-card memory cannot support it, multi-card inference is required.

#### Distributed Scenario

In distributed multi-card inference scenarios, it is necessary to enable parallel configurations in the settings and adjust the model parallel strategy. The configurations that need to be modified are as follows:

| Configuration Item | Configuration Description | Modification Method |
|----|----|--------|
|use_parallel |Parallel switch |Needs to be set to True during distributed inference|
|parallel_config |Parallel strategy |Currently, online inference only supports model parallelism. Set model_parallel to the number of cards used|

#### Scenarios with Other Video Memory Specifications

On devices without 64GB of video memory (on-chip memory), it is necessary to adjust the maximum video memory size occupied by MindSpore. The configurations that need to be modified are as follows:

| Configuration Item | Configuration Description | Modification Method |
|----|----|--------|
|max_device_memory|The maximum video memory that MindSpore can occupy|It is necessary to reserve part of the video memory for communication. Generally, devices with 64GB video memory are configured to be less than 60GB, and devices with 32GB video memory are configured to be less than 30GB. When the number of cards is relatively large, it may need to be reduced according to the actual situation.|

## Usage Example

Mindspore Transformers provides YAML configuration file templates for the Qwen3 series models [predict_qwen3.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml), Qwen3 models of different specifications can perform inference tasks using this template by modifying relevant configurations.

Taking Qwen3-32B as an example, the configuration that needs to be modified for reasoning YAML is as follows:

1. Modify pretrained_model_dir to the folder path of the model file of Qwen3-32B

    ```yaml
    pretrained_model_dir: "path/to/Qwen3-32B"
    ```

2. The Qwen3-32B requires at least 4 cards and the parallel configuration needs to be modified

    ```yaml
    use_parallel: True
    parallel_config:
        model_parallel: 4
    ```

Subsequent operations regarding the execution of reasoning tasks, please refer to [Qwen3's README](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/README.md#%E6%8E%A8%E7%90%86%E6%A0%B7%E4%BE%8B).

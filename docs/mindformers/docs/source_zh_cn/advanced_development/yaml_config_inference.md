# 推理配置模板使用指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/advanced_development/yaml_config_inference.md)

## 概述

当前Mcore架构模型在推理时，支持读取Hugging Face模型目录来实例化模型，因此MindSpore Transformers精简了模型的YAML配置文件，从原先每个模型、每个规格都有一份YAML，统一成一份YAML配置模板。不同规格模型在在线推理时，只需要套用配置模板，配置好从Hugging Face或ModelScope下载的模型目录，再修改少数必要配置，即可进行推理。

## 使用方法

使用推理配置模板进行推理时，需要根据实际情况，修改其中的部分配置。

### 必须修改的配置（Required）

配置模板不包含模型的配置，依赖读取Hugging Face或ModelScope的模型配置，来实例化模型。因此必须修改如下配置：

|配置项|配置说明|修改方法|
|----|----|--------|
|pretrained_model_dir|模型目录的路径|修改成从Hugging Face或ModelScope的下载的模型文件的文件夹路径|

### 可选的场景化配置（Optional）

以下不同使用场景需要对部分配置进行修改：

#### 默认场景（单卡、64GB显存）

推理配置模板默认为单卡64GB显存的场景配置，此时无需额外修改配置。需注意如果模型规模过大，单卡显存无法支持时，需要进行多卡推理。

#### 分布式场景

分布式的多卡推理场景需要在配置中打开并行配置，并调整模型并行策略，需要修改的配置如下：

|配置项|配置说明|修改方法|
|----|----|--------|
|use_parallel|并行开关|分布式推理时需要设置为True|
|parallel_config|并行策略|当前在线推理仅支持模型并行，设置model_parallel为使用的卡数|

#### 其他显存规格场景

非64GB显存（片上内存）的设备上，需要调整MindSpore占用的最大显存大小，需要修改的配置如下：

|配置项|配置说明|修改方法|
|----|----|--------|
|max_device_memory|MindSpore可占用的最大显存|需要为通信预留部分显存，一般情况下64GB显存的设备配置为<60GB，32GB显存的设备配置为<30GB。卡数比较多时可能还需根据实际情况减小。|

## 使用样例

Mindspore Transformers提供了Qwen3系列模型的YAML配置文件模板[predict_qwen3.yaml](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/predict_qwen3.yaml)，不同规格的Qwen3模型可以通过修改相关配置使用该模板执行推理任务。

以Qwen3-32B为例，按照如下步骤修改YAML配置文件：

1. 修改pretrained_model_dir为Qwen3-32B的模型文件的文件夹路径

    ```yaml
    pretrained_model_dir: "path/to/Qwen3-32B"
    ```

2. Qwen3-32B至少需要4卡，需要修改并行配置

    ```yaml
    use_parallel: True
    parallel_config:
        model_parallel: 4
    ```

关于执行推理任务的后续操作，详细可见[Qwen3的README](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/README.md#%E6%8E%A8%E7%90%86%E6%A0%B7%E4%BE%8B)。

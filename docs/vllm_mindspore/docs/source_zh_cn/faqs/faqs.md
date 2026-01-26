# 常见问题

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1.post1/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/r2.7.1.post1/docs/vllm_mindspore/docs/source_zh_cn/faqs/faqs.md)

## 安装相关问题

### 源码安装时报错`ModuleNotFoundError: No module named 'mindspore'`

- 错误关键信息：

    在执行以下命令安装vLLM-MindSpore Plugin

    ```bash
    git clone -b r0.5.0 https://atomgit.com/mindspore/vllm-mindspore.git
    cd vllm-mindspore
    bash install_depend_pkgs.sh
    pip install .
    ```

    得到如下报错信息

    ```text
    ModuleNotFoundError: No module named 'mindspore'
    ```

- 解决思路：

    1. 请检查是否已正确安装MindSpore。如果未安装，请参考[MindSpore安装指南](https://www.mindspore.cn/install)，或参考[安装指南](../getting_started/installation/installation.md)进行安装，确认`bash install_depend_pkgs.sh`已执行成功。
    2. 请检查`pip`版本是否大于等于25.3。如果是，则使用以下命令编译并安装vLLM-MindSpore Plugin：

        ```bash
        git clone -b r0.5.0 https://atomgit.com/mindspore/vllm-mindspore.git
        cd vllm-mindspore
        bash install_depend_pkgs.sh
        pip install --no-build-isolation .
        ```

## 模型相关问题

### git-lfs安装

1. 请到以下链接获取对应的[git-lfs安装包](https://github.com/git-lfs/git-lfs/releases/tag/v3.0.1)。
2. 下载并安装：

    ```bash
    mkdir git-lfs
    cd git-lfs
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-arm64-v3.0.1.tar.gz --no-check-certificate
    tar zxvf git-lfs-linux-arm64-v3.0.1.tar.gz
    bash install.sh
    ```

3. 校验是否安装成功：

    ```bash
    git lfs install
    ```

   若返回 `Git LFS initialized.`，则已安装成功。

## 部署相关问题

### 拉起在线推理时，报`aclnnNonzeroV2`相关错误

- 错误关键信息：

    ```text
    RuntimeError: Call aclnnNonzeroV2 failed, detail:E39999: Inner Error
    ```

- 解决思路：

    请检查CANN与MindSpore的配套关系是否正确。

### `import vllm_mindspore`时找不到`torch`

- 错误关键信息：

    ```text
    importlib.metadata.PackageNotFoundError: No package metadata was found for torch
    ```

- 解决思路：

    vLLM-MindSpore插件相关依赖未完整安装，如缺少`torch`、`MSAdapter`等组件。请参考[安装指南](../getting_started/installation/installation.md)进行安装。

### 推理时报`vllm._C`相关的告警

- 告警关键信息：

    ```text
    Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
    ```

- 说明：
    该告警为非影响推理的告警，不影响模型的离线推理。

### 推理过程中报内存不足相关的问题

- 错误关键信息：
    出现关键信息`Out of Memory`，或出现`Allocate memory failed`，则为设备内存不足的问题。

- 解决思路：
    该报错表示设备内存不足，可能由多种原因导致，建议按以下方面排查：

    1. 使用命令`npu-smi info`，确认卡是否独占状态。若不是独占状态，可尝试将卡设置为独占状态。
    2. 确认模型参数是否过大，导致内存不足。若模型参数过大，可尝试减少模型参数，或使用分布式推理。
    3. 若使用在线推理，可以调整`--max-model-len`参数，减少模型最大长度，减少内存占用；或提高`--gpu-memory-utilization`，从而提高显存利用率。
    4. 若使用离线推理，可以在初始化`LLM`对象时，对`max_model_len`参数进行设置，减少模型最大长度；或对`gpu_memory_utilization`参数进行提高，增加显存使用率。
    5. 调整混合并行策略，适当增大流水线并行（pp）和模型并行（mp），并相应减小数据并行（dp），保持`dp * mp * pp = device_num`，必要时增加NPU数量。

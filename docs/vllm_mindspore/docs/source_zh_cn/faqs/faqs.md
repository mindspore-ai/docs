# 常见问题

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/faqs/faqs.md)

## 安装相关问题

### 源码安装时报错`ModuleNotFoundError: No module named 'mindspore'`

- 错误关键信息：

    在执行以下命令安装vLLM-MindSpore Plugin

    ```bash
    git clone https://gitee.com/mindspore/vllm-mindspore.git
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
        git clone https://gitee.com/mindspore/vllm-mindspore.git
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

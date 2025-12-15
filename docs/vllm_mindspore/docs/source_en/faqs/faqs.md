# Frequently Asked Questions

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/faqs/faqs.md)

## Installation-related Issues

### Source Installation Error: `ModuleNotFoundError: No module named 'mindspore'`

- Key error message:

    Execute the following command to install vLLM-MindSpore Plugin:

    ```bash
    git clone https://gitee.com/mindspore/vllm-mindspore.git
    cd vllm-mindspore
    bash install_depend_pkgs.sh
    pip install .
    ```

    But get the following error message:

    ```text
    ModuleNotFoundError: No module named 'mindspore'
    ```

- Solution:

    1. Please check if MindSpore is installed correctly. If not, please refer to the [MindSpore installation guide](https://www.mindspore.cn/install/en/) or [installation guide](../getting_started/installation/installation.md) for installation, and confirm that `bash install_depend_pkgs.sh` has been executed successfully.
    2. Please check if the `pip` version is greater than or equal to 25.3. If so, please use the following command to compile and install vLLM-MindSpore Plugin:

        ```bash
        git clone https://gitee.com/mindspore/vllm-mindspore.git
        cd vllm-mindspore
        bash install_depend_pkgs.sh
        pip install --no-build-isolation .
        ```

## Model-related Issues

### Git-LFS Installation

1. Obtain the corresponding [git-lfs installation package](https://github.com/git-lfs/git-lfs/releases/tag/v3.0.1) from the following link.
2. Download and install:

    ```bash
    mkdir git-lfs
    cd git-lfs
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-arm64-v3.0.1.tar.gz --no-check-certificate
    tar zxvf git-lfs-linux-arm64-v3.0.1.tar.gz
    bash install.sh
    ```

3. Verify successful installation:

    ```bash
    git lfs install
    ```

   If `Git LFS initialized.` is returned, the installation was successful.

## Deployment-related Issues

### `aclnnNonzeroV2` Related Error When Starting Online Inference

- Key error message:

    ```text
    RuntimeError: Call aclnnNonzeroV2 failed, detail:E39999: Inner Error
    ```

- Solution:
    Check whether the CANN and MindSpore versions are correctly matched.

### `torch` Not Found When Importing `vllm_mindspore`

- Key error message:

    ```text
    importlib.metadata.PackageNotFoundError: No package metadata was found for torch
    ```

- Solution:

    vLLM-MindSpore Plugin related dependencies are not installed completely, such as missing `torch`, `MSAdapter` and other components. Please refer to the [installation guide](../getting_started/installation/installation.md) for installation.

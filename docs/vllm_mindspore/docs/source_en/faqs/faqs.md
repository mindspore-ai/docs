# Frequently Asked Questions

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1.post1/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.7.1.post1/docs/vllm_mindspore/docs/source_en/faqs/faqs.md)

## Installation-related Issues

### Source Installation Error: `ModuleNotFoundError: No module named 'mindspore'`

- Key error message:

    Execute the following command to install vLLM-MindSpore Plugin:

    ```bash
    git clone -b r0.5.0 https://atomgit.com/mindspore/vllm-mindspore.git
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
        git clone -b r0.5.0 https://atomgit.com/mindspore/vllm-mindspore.git
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

### Inference Warning Related to `vllm._C`

- Key warning message:

    ```text
    Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
    ```

- Description:
    This warning does not affect inference and does not impact the offline inference of the model.

### Out of Memory During Inference

- Key error message:
    If the key message `Out of Memory` or `Allocate memory failed` appears, it indicates insufficient device memory.

- Solution:
    This error indicates that the device memory is insufficient and may be caused by several factors. It is recommended to investigate the following aspects:

    1. Run the command `npu-smi info` to check whether the card is in exclusive mode. If not, try setting the card to exclusive mode.
    2. Verify whether the model parameters are too large, leading to insufficient memory. If so, try reducing the model parameters or using distributed inference.
    3. For online inference, adjust the `--max-model-len` parameter to reduce the maximum model length and lower memory usage, or increase `--gpu-memory-utilization` to improve GPU memory utilization.
    4. For offline inference, when initializing the `LLM` object, set the `max_model_len` parameter to reduce the maximum model length, or increase the `gpu_memory_utilization` parameter to raise GPU memory usage.
    5. Tune the hybrid parallelism strategy by appropriately increasing pipeline parallelism (pp) and model parallelism (mp) while correspondingly decreasing data parallelism (dp), ensuring `dp * mp * pp = device_num`. Increase the number of NPUs if necessary.

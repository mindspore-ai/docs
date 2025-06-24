# Frequently Asked Questions

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/faqs/faqs.md)

## Model-related Issues

### Git-LFS Installation

1. Obtain the corresponding [git-lfs installation package](https://github.com/git-lfs/git-lfs/releases/tag/v3.0.1) from the following link.
2. Download and install:

   ```shell
   mkdir git-lfs
   cd git-lfs
   wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-arm64-v3.0.1.tar.gz --no-check-certificate
   tar zxvf git-lfs-linux-arm64-v3.0.1.tar.gz
   bash install.sh
   ```

3. Verify successful installation:

   ```shell
   git lfs install
   ```

   If `Git LFS initialized.` is returned, the installation was successful.

## Deployment-related Issues

### Model Fails to Load During Offline/Online Inference

- Key error message:

   ```text
   raise ValueError(f"{config.load_checkpoint} is not a valid path to load checkpoint ")
   ```

- Solution:
  1. Check if the model path exists and is valid;
  2. If the model path exists and the model files are in `safetensors` format, confirm whether the yaml file contains the `load_ckpt_format: "safetensors"` field:
     1. Print the path of the yaml file used by the model:

        ```bash
        echo $MINDFORMERS_MODEL_CONFIG
        ```

     2. Check the yaml file. If the `load_ckpt_format` field is missing, add it:

        ```text
        load_ckpt_format: "safetensors"
        ```

### `aclnnNonzeroV2` Related Error When Starting Online Service

- Key error message:

   ```text
   RuntimeError: Call aclnnNonzeroV2 failed, detail:E39999: Inner Error
   ```

   Check whether the CANN and MindSpore versions are correctly matched.

### `torch` Not Found When Importing `vllm_mindspore`

- Key error message:

   ```text
   importlib.metadata.PackageNotFoundError: No package metadata was found for torch
   ```

   Execute the following commands to uninstall torch-related components:

   ```bash
   pip uninstall torch
   pip uninstall torchvision
   ```

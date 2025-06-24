# 常见问题

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/faqs/faqs.md)

## 模型相关问题

### git-lfs安装

1. 请到以下链接获取对应的[git-lfs安装包](https://github.com/git-lfs/git-lfs/releases/tag/v3.0.1)。
2. 下载并安装：

   ```shell
   mkdir git-lfs
   cd git-lfs
   wget https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-arm64-v3.0.1.tar.gz --no-check-certificate
   tar zxvf git-lfs-linux-arm64-v3.0.1.tar.gz
   bash install.sh
   ```

3. 校验是否安装成功：

   ```shell
   git lfs install
   ```

   若返回 `Git LFS initialized.`，则已安装成功。

## 部署相关问题

### 离线或在线推理时，报模型无法加载

- 错误关键信息：

   ```text
   raise ValueError(f"{config.load_checkpoint} is not a valid path to load checkpoint ")
   ```

- 解决思路：
  1. 检查模型路径是否存在且合法；
  2. 若模型路径存在，且其中的模型文件为`safetensors`格式，则需要确认yaml文件中，是否已含有`load_ckpt_format: "safetensors"`字段；
     1. 打印模型所使用的yaml文件路径：

        ```bash
        echo $MINDFORMERS_MODEL_CONFIG
        ```

     2. 查看该yaml文件，若不存在`load_ckpt_format`字段，则添加该字段：

        ```text
        load_ckpt_format: "safetensors"
        ```

### 拉起在线服务时，报`aclnnNonzeroV2`相关错误

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
   请执行以下命令，卸载torch相关组件：

   ```bash
   pip uninstall torch
   pip uninstall torchvision
   ```

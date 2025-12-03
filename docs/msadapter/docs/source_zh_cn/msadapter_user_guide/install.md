# 安装

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/msadapter/docs/source_zh_cn/msadapter_user_guide/install.md)

在昇腾NPU设备上，完成[昇腾固件](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)的安装后，执行以下步骤完成PyTorch、MindSpore和MSAdapter的安装：

## 1. 安装PyTorch和MindSpore

```bash
pip install torch==2.1.0
pip install mindspore
```

## 2. 下载安装MSAdapter源码

目前MSAdapter不支持`pip install msadapter`方式安装，仅支持源码直接使用和源码编译安装。

- 如果用户希望直接使用源码，设置如下环境环境变量：

    ``` bash
    export PYTHONPATH=${your_workspace}/msadapter/:$PYTHONPATH
    export PYTHONPATH=${your_workspace}/msadapter/msa_thirdparty:$PYTHONPATH
    ```

    其中，your_workspace是git clone下载的目录。此方法不会影响用户的PyTorch使用。

- 如果用户希望以Python安装包编译的形式使用，进入MSAdapter目录，进行源码编译操作：

    ```bash
    git clone https://gitee.com/mindspore/msadapter.git
    cd msadapter
    bash scripts/build.sh
    pip install ${your_workspace}/msadapter/dist/*.whl
    export PYTHONPATH=/*/site-packages/msa_thirdparty:$PYTHONPATH
    # /*/site-packages 指python环境下的安装包路径，可以使用pip show msadapter获取。
    ```

    直接安装会覆盖原始PyTorch的使用，如果希望同时使用PyTorch和MSAdapter，可以考虑直接使用源码。

安装完成后，PyTorch的实际执行将替换为MSAdapter，后端则为MindSpore动态图模式。

# 模型加密保护

`Linux` `Ascend` `GPU` `CPU` `模型保护` `企业` `高级`

<!-- TOC -->

- [模型加密保护](#模型加密保护)
    - [概述](#概述)
    - [安全导出CheckPoint文件](#安全导出CheckPoint文件)
    - [加载密文CheckPoint文件](#加载密文CheckPoint文件)

<!-- TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/training/source_zh_cn/advanced_use/model_encrypt_protection.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 概述

MindSpore框架提供通过加密对模型文件进行保护的功能，使用对称加密算法对参数文件或推理模型进行加密，使用时直接加载密文模型完成推理或增量训练。
目前加密方案支持在Linux平台下对CheckPoint参数文件的保护。

以下通过示例来介绍CheckPoint文件的加密保存和读取的方法。

> 你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/docs/blob/master/tutorials/tutorial_code/model_encrypt_protection/encrypt_checkpoint.py>

## 安全导出CheckPoint文件

目前MindSpore支持使用Callback机制传入回调函数`ModelCheckpoint`对象以保存模型参数，用户可以通过配置`CheckpointConfig`对象来启用参数文件的加密保护。具体配置方法如下：

```python
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10, enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
ckpoint_cb = ModelCheckpoint(prefix='lenet_enc', directory=None, config=config_ck)
model.train(10, dataset, callbacks=ckpoint_cb)
```

上述代码中，通过在`CheckpointConfig`中初始化加密密钥和加密模式来启用模型加密。

- `enc_key`表示用于对称加密的密钥。

- `enc_mode`表示使用哪种加密模式。

除了上面这种保存模型参数的方法，还可以调用`save_checkpoint`接口来保存模型参数，使用方法如下：

```python
from mindspore import save_checkpoint

save_checkpoint(network, 'lenet_enc.ckpt', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

其中`enc_key`和`enc_mode`的定义同上。

## 加载密文CheckPoint文件

MindSpore提供`load_checkpoint`和`load_distributed_checkpoint`分别用于单文件和分布式场景下加载CheckPoint参数文件。以单文件场景为例，可以用如下方式加载密文CheckPoint文件：

```python
from mindspore import load_checkpoint

param_dict = load_checkpoint('lenet_enc.ckpt', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

上述代码中，通过指定`dec_key`和`dec_mode`来启用对密文文件的读取。

- `dec_key`表示用于对称解密的密钥。

- `dec_mode`表示使用哪种解密模式。

分布式场景的方式类似，在调用`load_distributed_checkpoint`时指定`dec_key`和`dec_mode`即可。

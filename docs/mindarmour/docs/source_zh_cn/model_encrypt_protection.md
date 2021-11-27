# 模型加密保护

<!-- TOC -->

- [模型加密保护](#模型加密保护)
    - [概述](#概述)
    - [安全导出CheckPoint文件](#安全导出CheckPoint文件)
    - [加载密文CheckPoint文件](#加载密文CheckPoint文件)
    - [安全导出MindIR文件](#安全导出MindIR文件)
    - [加载密文MindIR文件](#加载密文MindIR文件)
    - [端侧模型保护](#端侧模型保护)
        - [模型转换工具](#模型转换工具)

<!-- TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindarmour/docs/source_zh_cn/model_encrypt_protection.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/notebook/mindspore_model_encrypt_protection.ipynb"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbWFzdGVyL25vdGVib29rL21pbmRzcG9yZV9tb2RlbF9lbmNyeXB0X3Byb3RlY3Rpb24uaXB5bmI=&imageid=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_modelarts.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/notebook/mindspore_model_encrypt_protection.py"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_download_code.png"></a>

## 概述

MindSpore框架提供通过加密对模型文件进行保护的功能，使用对称加密算法对参数文件或推理模型进行加密，使用时直接加载密文模型完成推理或增量训练。
目前加密方案支持在Linux平台下对CheckPoint和MindIR模型文件的保护。

以下通过示例来介绍加密导出和解密加载的方法。

> 你可以在这里下载完整的样例代码：<https://gitee.com/mindspore/docs/blob/master/docs/sample_code/model_encrypt_protection/encrypt_checkpoint.py>

## 安全导出CheckPoint文件

目前MindSpore支持用Callback机制在训练过程中保存模型参数，用户可以在`CheckpointConfig`对象中配置加密密钥和加密模式，并将其传入`ModelCheckpoint`来启用参数文件的加密保护。具体配置方法如下：

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

## 安全导出MindIR文件

MindSpore提供的`export`接口可导出MindIR、AIR、ONNX等格式的模型，在导出MindIR模型时可用如下方式启用加密保护：

```python
from mindspore import export
input_arr = Tensor(np.zeros([32, 3, 32, 32], np.float32))
export(network, input_arr, file_name='lenet_enc', file_format='MINDIR', enc_key=b'0123456789ABCDEF', enc_mode='AES-GCM')
```

> AIR和ONNX格式暂不支持加密保护。

## 加载密文MindIR文件

云侧使用Python编写脚本，可以用`load`接口加载MindIR模型，在加载密文MindIR时，通过指定`dec_key`和`dec_mode`对模型进行解密。

```python
from mindspore import load
graph = load('lenet_enc.mindir', dec_key=b'0123456789ABCDEF', dec_mode='AES-GCM')
```

对于C++脚本，MindSpore也提供了`Load`接口以加载MindIR模型，接口定义可参考[api文档](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html?highlight=load)：

在加载密文模型时，通过指定`dec_key`和`dec_mode`对模型进行解密。

```C++
#include "include/api/serialization.h"

namespace mindspore{
  Graph graph;
  const unsigned char[] key = "0123456789ABCDEF";
  const size_t key_len = 16;
  Key dec_key(key, key_len);
  Serialization::Load("./lenet_enc.mindir", ModelType::kMindIR, &graph, dec_key, "AES-GCM");
} // namespace mindspore
```

## 端侧模型保护

### 模型转换工具

MindSpoer Lite提供的模型转换工具conveter可以将密文的mindir模型转化为明文ms模型，用户只需在调用该工具时指明密钥和解密模式即可，注意这里的密钥为十六进制表示的字符串，如前面定义的`b'0123456789ABCDEF'`对应的十六进制表示为`30313233343536373839414243444546`，Linux平台用户可以使用`xxd`工具对字节表示的密钥进行十六进制表达转换。具体调用方法如下：

```shell
./converter_tools --fmk=MINDIR --modelFile=./lenet_enc.mindir --outputFile=lenet --decryptKey=30313233343536373839414243444546 --decryptMode=AES-GCM
```


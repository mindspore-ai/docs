# 比较与torch.cos的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cos.md)

## torch.cos

```text
torch.cos(input, *, out=None) -> Tensor
```

更多内容详见[torch.cos](https://pytorch.org/docs/1.8.1/generated/torch.cos.html)。

## mindspore.ops.cos

```text
mindspore.ops.cos(input) -> Tensor
```

更多内容详见[mindspore.ops.cos](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.cos.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch：参数 `input` 的dtype可以是 ``complex``，``int`` 或 ``float`` 。

MindSpore：参数 `input` 的dtype可以是 ``float16`` ，``float32`` ，``float64`` ，``complex64`` 或 ``complex128`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input | input | 二者均为Tensor，torch.cos的参数 `input` 的dtype可以是 ``complex``，``int`` 或 ``float`` ；mindspore.ops.cos的参数 `input` 的dtype可以是 ``float16`` ，``float32`` ，``float64`` ，``complex64`` 或 ``complex128`` 。|
|      | 参数2 | out | - | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.2/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

# 比较与torch.polygamma的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/polygamma.md)

## torch.polygamma

```text
torch.polygamma(n, input, *, out=None) -> Tensor
```

更多内容详见[torch.polygamma](https://pytorch.org/docs/1.8.1/generated/torch.polygamma.html)。

## mindspore.ops.polygamma

```text
mindspore.ops.polygamma(n, input) -> Tensor
```

更多内容详见[mindspore.ops.polygamma](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.polygamma.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch：参数 `input` 的dtype可以是 ``int`` 或``float`` 。

MindSpore：参数 `input` 的dtype可以是 ``float16`` ，``float32`` 或 ``float64`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | n | n | - |
|      | 参数2 | input | input | 二者均为Tensor，torch.polygamma的参数 `input` 的dtype可以是 ``int`` 或 ``float`` ，mindspore.ops.polygamma的参数 `input` 的dtype可以是 ``float16`` ，``float32`` 或 ``float64`` 。|
|      | 参数3 | out | - | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

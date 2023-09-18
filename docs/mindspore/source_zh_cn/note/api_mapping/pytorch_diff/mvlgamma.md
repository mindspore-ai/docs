# 比较与torch.mvlgamma的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mvlgamma.md)

## torch.mvlgamma

```text
torch.mvlgamma(input, p) -> Tensor
```

更多内容详见[torch.mvlgamma](https://pytorch.org/docs/1.8.1/generated/torch.mvlgamma.html)。

## mindspore.ops.mvlgamma

```text
mindspore.ops.mvlgamma(input, p) -> Tensor
```

更多内容详见[mindspore.ops.mvlgamma](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.mvlgamma.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch：参数 `input` 的dtype可以是 ``int`` 或 ``float`` 。

MindSpore：参数 `input` 的dtype可以是 ``float32`` 或 ``float64`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input | input | 二者均为Tensor，torch.mvlgamma的参数 `input` 的dtype可以是 ``int`` 或 ``float`` ，mindspore.ops.mvlgamma的参数 `input` 的dtype可以是 ``float32`` 或 ``float64`` 。|
|      | 参数2 | p | p | - |

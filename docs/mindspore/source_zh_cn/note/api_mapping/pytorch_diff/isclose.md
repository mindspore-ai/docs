# 比较与torch.isclose的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/isclose.md)

## torch.isclose

```text
torch.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
```

更多内容详见[torch.isclose](https://pytorch.org/docs/1.8.1/generated/torch.isclose.html)。

## mindspore.ops.isclose

```text
mindspore.ops.isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
```

更多内容详见[mindspore.ops.isclose](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.isclose.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch: 参数 `input` 和 `other` 的dtype可以是 ``bool``，``int`` 和 ``float`` 。

MindSpore： 参数 `x1` 和 `x2` 的dtype可以是 ``int32`` ， ``float32`` 和 ``float16`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x1 | 二者参数名不同。二者均为Tensor，但参数 `input` 的dtype可以是 ``bool`` ，``int`` 和 ``float`` ，参数 `x1` 的dtype可以是 ``int32`` ，``float32`` 和 ``float16`` 。|
|  | 参数2 | other | x2 | 二者参数名不同。二者均为Tensor，但参数 `other` 的dtype可以是 ``bool`` ， ``int`` 和 ``float`` ，参数 `x2` 的dtype可以是 ``int32`` ，``float32`` 和 ``float16`` 。|
|  | 参数3 | rtol | rtol | - |
|  | 参数4 | atol | atol | - |
|  | 参数5 | equal_nan | equal_nan | - |

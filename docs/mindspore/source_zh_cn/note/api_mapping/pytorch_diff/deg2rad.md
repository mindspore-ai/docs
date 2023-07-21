# 比较与torch.deg2rad的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/deg2rad.md)

## torch.deg2rad

```text
torch.deg2rad(input, *, out=None) -> Tensor
```

更多内容详见[torch.deg2rad](https://pytorch.org/docs/1.8.1/generated/torch.deg2rad.html)。

## mindspore.ops.deg2rad

```text
mindspore.ops.deg2rad(x) -> Tensor
```

更多内容详见[mindspore.ops.deg2rad](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.deg2rad.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch：参数 `input` 的dtype可以是 ``int`` 或 ``float`` 。

MindSpore： 参数 `x` 的dtype可以是 ``float16`` ，``float32`` 或 ``float64`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input | x | 二者参数名不同。二者均为Tensor，但参数 `input` 的dtype可以是 ``int`` 或 ``float`` ，参数 `x` 的dtype可以是 ``float16`` ，``float32`` 或 ``float64`` 。|
|      | 参数2 | out | - | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

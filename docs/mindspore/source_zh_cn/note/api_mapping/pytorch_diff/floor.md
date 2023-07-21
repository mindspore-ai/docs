# 比较与torch.floor的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/floor.md)

## torch.floor

```text
torch.floor(input, *, out=None) -> Tensor
```

更多内容详见[torch.floor](https://pytorch.org/docs/1.8.1/generated/torch.floor.html)。

## mindspore.ops.floor

```text
mindspore.ops.floor(input) -> Tensor
```

更多内容详见[mindspore.ops.floor](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.floor.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch：参数 `input` 的dtype可以是 ``int`` 和 ``float`` 。

MindSpore：参数 `input` 的dtype可以是 ``float16`` ，``float32`` 和 ``float64`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input | input | 二者均为Tensor，torch.floor的参数 `input` 的dtype可以是 ``int`` 和 ``float`` ，mindspore.ops.floor的参数 `input` 的dtype可以是 ``float16`` ，``float32`` 和 ``float64`` 。|
|      | 参数2 | out | - | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

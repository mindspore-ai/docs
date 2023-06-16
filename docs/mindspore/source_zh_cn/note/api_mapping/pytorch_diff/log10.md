# 比较与torch.log10的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/log10.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.log10

```text
torch.log10(input, *, out=None) -> Tensor
```

更多内容详见[torch.log10](https://pytorch.org/docs/1.8.1/generated/torch.log10.html)。

## mindspore.ops.log10

```text
mindspore.ops.log10(input) -> Tensor
```

更多内容详见[mindspore.ops.log10](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.log10.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch: 参数 `input` 的dtype可以是 ``int`` 或 ``float`` 。

MindSpore： 在GPU和CPU平台上，参数 `input` 的dtype可以是 ``float16`` ， ``float32`` 或 ``float64`` ；在Ascend平台上，参数 `input` 的dtype可以是 ``float16`` 或 ``float32`` 。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input | input | 二者均为Tensor，torch.log10的参数 `input` 的dtype可以是 ``int`` 或 ``float`` ，mindspore.ops.log10的参数 `input` 的dtype在GPU和CPU平台上可以是 ``float16`` ， ``float32`` 或 ``float64`` ， 在Ascend平台上可以是 ``float16`` 或 ``float32`` 。|

# 比较与torch.torch_op的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/examples/api_mapping_with_diffs_template.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.torch_op

```text
torch.torch_op(torch_arg1, torch_arg2, torch_agr3, torch_agr4) -> Tensor
```

更多内容详见 附上torch.torch_op官网链接。

## mindspore.ops.ms_op

```text
mindspore.ops.ms_op(ms_arg1, ms_arg2,  ms_agr3) -> Tensor
```

更多内容详见 附上mindspore.ops.ms_op官网链接。

## 差异对比

PyTorch：torch_op算子功能简要描述。

MindSpore: ms_op算子功能简要描述， 需要重点突出与PyTorch的区别。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | torch_arg1 | ms_arg1 |功能一致，参数名不同 |
| | 参数2 | torch_arg2 | ms_arg2 | 功能一致，参数名不同， 默认值不同 |
| | 参数3 | torch_arg3 | ms_arg3 |功能一致，参数传入方式不同 |
| | 参数4 | torch_arg4 | - |描述PyTorch上torch_arg4实现的功能，附上“MindSpore无此参数” |
| | 参数5 | torch_arg5 | ms_arg4 |-|

### 代码示例1

说明：指出默认值各为多少， 然后说明默认值不同对功能和使用上的影响， 如何调整MindSpore能得到和PyTorch一样的结果。

```python
PyTorch example代码1
MindSpore example代码1
```

### 代码示例2

说明：指出分别是怎样传入的，说明是否会影响功能， 如果会，如何调整MindSpore能得到和PyTorch一样的结果。

```python
PyTorch example代码2
MindSpore example代码2
```

### 代码示例3

说明：描述PyTorch上该参数功能，如果在MindSpore上可以使用其他接口/算子组合实现， 给出实现过程， 如果不能，提供分析报告，说明原因。

```python
PyTorch example代码3
MindSpore example代码3
```

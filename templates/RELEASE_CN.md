# MindSpore Release Notes

[View English](./RELEASE.md)

## MindSpore 2.0.0 Release Notes

### 主要特性及增强

#### PyNative

- [STABLE] MindSpore默认模式切换成PyNative模式。
- [BETA] 完成动态shape执行方案重构，提升反向构图性能，支持非padding方案的动态shape网络编程，当前主要验证网络Transformer-GPU、YOLOV5-GPU、ASR-Ascend。

#### Executor

- [DEMO] 在动态图模式中加入分布式训练。

### API变更

#### 新增/增强API

##### Python APIs

- 新增nn接口 `nn.AdaptiveAvgPool3d` 。
- 新增ops functional接口 `ops.addcdiv` 。
- `ops.approximate_equal` 新增GPU、CPU支持。

##### C++ APIs

#### 非兼容性变更

##### Python APIs

- `nn.EmbeddingLookup` 新增参数 `sparse` ，设置是否使用稀疏模式。[(!8202)](https://gitee.com/mindspore/mindspore/pulls/8202)

  <table>
  <tr>
  <td style="text-align:center"> 1.10.0 </td> <td style="text-align:center"> 2.0.0 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  >>> from mindspore.nn import EmbeddingLookup
  >>>
  >>> input_indics = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
  >>> result = EmbeddingLookup(4,2)(input_indices)
  >>> print(result.shape)
  (2, 2, 2)
  </code></pre>
  </td>
  <td><pre style="display: block;"><code class="language-python">
  >>> from mindspore.nn import EmbeddingLookup
  >>>
  >>> input_indics = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
  >>> result = EmbeddingLookup(4,2)(input_indices, sparse=False)
  >>> print(result.shape)
  (2, 2, 2)
  </code></pre>
  </td>
  </tr>
  </table>

- `nn.probability.bijector.GumbelCDF` 移除参数 `dtype` 。[(!8191)](https://gitee.com/mindspore/mindspore/pulls/8191)

  <table>
  <tr>
  <td style="text-align:center"> 1.10.0 </td> <td style="text-align:center"> 2.0.0 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  >>> import nn.probability.bijector as msb
  >>> from mindspore import stype as mstype
  >>>
  >>> bijector = msb.GumbelCDF(loc=0.0, scale=1.0, dtype=mstype.float32)
  </code></pre>
  </td>
  <td><pre style="display: block;"><code class="language-python">
  >>> import nn.probability.bijector as msb
  >>> from mindspore import stype as mstype
  >>>
  >>> bijector = msb.GumbelCDF(loc=0.0, scale=1.0)
  </code></pre>
  </td>
  </tr>
  </table>

- `nn.layer.combined.Conv2dBnAct` 从 nn.layer.quant 移动到 nn.layer.combined 。[(!8187)](https://gitee.com/mindspore/mindspore/pulls/8187)

  <table>
  <tr>
  <td style="text-align:center"> 1.10.0 </td> <td style="text-align:center"> 2.0.0 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  >>> from mindspore.nn.layer.quant import Conv2dBnAct
  </code></pre>
  </td>
  <td><pre style="display: block;"><code class="language-python">
  >>> from mindspore.nn.layer.combined import Conv2dBnAct
  </code></pre>
  </td>
  </tr>
  </table>

- `mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg` 接口废弃使用。
- `ops.SparseApplyAdagrad` 接口废弃使用，可使用 `ops.SparseApplyAdagradV2` 接口替代。

##### C++ APIs

### 问题修复

#### Executor

- 修复图模式和PyNative模式之间共享权重是的执行错误。[(!26635)](https://gitee.com/mindspore/mindspore/pulls/26635)
- 修复了PyNative模式下释放内存时的概率问题。[(!25472)](https://gitee.com/mindspore/mindspore/pulls/25472)

#### Dataset

- 修复长时间运行数据集时内存异常增长的问题。[(!26237)](https://gitee.com/mindspore/mindspore/pulls/26237)
- 修复在Windows上使用中文路径保存MindRecord文件的问题。[(!28378)](https://gitee.com/mindspore/mindspore/pulls/28378)

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 2.0.0 Release Notes

### 主要特性及增强

……

### API变更

……

### 问题修复

……

### 贡献者

……

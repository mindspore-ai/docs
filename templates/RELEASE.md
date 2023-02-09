# MindSpore Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore 2.0.0 Release Notes

### Major Features and Improvements

#### PyNative

- [STABLE] The default mode of MindSpore is switched to PyNative.
- [BETA] Support dynamic shape without padding, three networks are supported as demos: Transformer-GPU, YOLOV5-GPU, ASR-Ascend.

#### Executor

- [DEMO] Add distributed training in PyNative mode.

### API Change

#### New APIs & Enhanced APIs

##### Python APIs

- Add nn API `nn.AdaptiveAvgPool3d` .
- Add ops functional API `ops.addcdiv` .
- `ops.approximate_equal` add GPU, CPU support.

##### C++ APIs

#### Incompatible Modification

##### Python APIs

- `nn.EmbeddingLookup` add the parameter `sparse` ，set to use sparse mode. [(!8202)](https://gitee.com/mindspore/mindspore/pulls/8202)

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

- `nn.probability.bijector.GumbelCDF` delete the parameter `dtype` . [(!8191)](https://gitee.com/mindspore/mindspore/pulls/8191)

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

- `nn.layer.combined.Conv2dBnAct` move from nn.layer.quant to nn.layer.combined . [(!8187)](https://gitee.com/mindspore/mindspore/pulls/8187)

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

- Delete `mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg` .
- Delete `ops.SparseApplyAdagrad` and use `ops.SparseApplyAdagradV2` instead.

##### C++ APIs

### Bug Fixes

#### Executor

- Fix the execution error when the weights are shared between graph mode and PyNative mode. [(!26635)](https://gitee.com/mindspore/mindspore/pulls/26635)
- Fixed the probability coredump when free memory under PyNative mode. [(!25472)](https://gitee.com/mindspore/mindspore/pulls/25472)

#### Dataset

- Fix memory increase abnormally when running dataset for a long time. [(!26237)](https://gitee.com/mindspore/mindspore/pulls/26237)
- Fix saving MindRecord files with Chinese path on Windows. [(!28378)](https://gitee.com/mindspore/mindspore/pulls/28378)

### Contributors

Thanks goes to these wonderful people:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy.

Contributions of any kind are welcome!

## MindSpore Lite 2.0.0 Release Notes

### Major Features and Improvements

...

### API Change

...

### Bug Fixes

...

### Contributors

...

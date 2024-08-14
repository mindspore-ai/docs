# Precision Tuning

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/acc_debug.md)

## FAQs and Solutions

- The following common problems may be encountered during the accuracy commissioning phase:
    - The first loss and the benchmark are not aligned:
         It means that the network positive and benchmark are not aligned, you can fix the network input, turn off randomness such as shuffle, save the output as npy at some key nodes of the network, and with the help of [TroubleShooter to see if the two sets of Tensor values (npy files) are equal or not](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/migrator.md#%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF4%E6%AF%94%E8%BE%83%E4%B8%A4%E7%BB84tensor%E5%80%BCnpy%E6%96%87%E4%BB%B6%E6%98%AF%E5%90%A6%E7%9B%B8%E7%AD%89), locate the first inconsistent position, and then bisect the position to analyze the positive where the difference leads to the loss and the benchmark misaligned to cause the accuracy problem.
    - The first loss is aligned with the benchmark, and subsequent losses are misaligned:
         The problem is mainly caused by the network reverse. This can be done with the help of [TroubleShooter comparing MindSpore to PyTorch ckpt/pth](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/migrator.md#%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF2%E6%AF%94%E5%AF%B9mindspore%E4%B8%8Epytorch%E7%9A%84ckptpth) to check the results of the network reverse update by comparing the values of the corresponding parameters of ckpt and pth.
    - Loss appears NAN/INF:
         [TroubleShooter obtains INF/NAN value throw points](https://gitee.com/mindspore/toolkits/blob/master/troubleshooter/docs/tracker.md#%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF2%E8%8E%B7%E5%8F%96infnan%E5%80%BC%E6%8A%9B%E5%87%BA%E7%82%B9) is used to identify the first location in the network where a NAN or INF appears.
         Overflow operator detection is also available via the [Dump](https://www.mindspore.cn/docs/en/master/model_train/debug/dump.html) tool.

## Accuracy Debugging Process

The accuracy debugging process is as follows:

### 1. Checking Parameters

This part includes checking all parameters and the number of trainable parameters, and checking the shape of all parameters.

- `Parameter` is used for PyTorch trainable parameters, and `requires_grad=False` or `buffer` is used for PyTorch untrainable parameters.
- `Parameter` is used for MindSpore trainable and untrainable parameters.
- The parameters of MindSpore and PyTorch are similar except BatchNorm. Note that MindSpore does not have parameters corresponding to `num_batches_tracked`. You can replace this parameter with `global_step` in the optimizer.

  | MindSpore | PyTorch |
  | --------- | --------|
  | gamma | weight |
  | beta | bias |
  | moving_mean | running_mean |
  | moving_variance | running_var |
  | -| num_batches_tracked |

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> Obtaining PyTorch Parameters </td> <td style="text-align:center"> Obtaining MindSpore Parameters </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
from torch import nn

class ptNet(nn.Module):
    def __init__(self):
          super(ptNet, self).__init__()
          self.fc = nn.Linear(1, 1)
    def construct(self, x):
        output = self.fc(x)
        return output

ptnet = ptNet()
all_parameter = []
trainable_params = []
# Obtain network parameters.
for name, item in ptnet.named_parameters():
    if item.requires_grad:
        trainable_params.append(item)
    all_parameter.append(item)
    print(name, item.shape)

for name, buffer in ptnet.named_buffers():
    all_parameter.append(buffer)
    print(name, buffer.shape)
print(f"all parameter numbers: {len(all_parameter)}")
print(f"trainable parameter numbers: {len(trainable_params)}")
```

Outputs:

```text
fc.weight torch.Size([1, 1])
fc.bias torch.Size([1])
all parameter numbers: 2
trainable parameter numbers: 2
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
from mindspore import nn

class msNet(nn.Cell):
    def __init__(self):
        super(msNet, self).__init__()
        self.fc = nn.Dense(1, 1, weight_init='normal')
    def construct(self, x):
        output = self.fc(x)
        return output

msnet = msNet()
# Obtain all parameters.
all_parameter = []
for item in msnet.get_parameters():
    all_parameter.append(item)
    print(item.name, item.data.shape)
print(f"all parameter numbers: {len(all_parameter)}")

# Obtain trainable parameters.
trainable_params = msnet.trainable_params()
for item in trainable_params:
    print(item.name, item.data.shape)
print(f"trainable parameter numbers: {len(trainable_params)}")
```

Outputs:

```text
fc.weight (1, 1)
fc.bias (1,)
all parameter numbers: 2
fc.weight (1, 1)
fc.bias (1,)
trainable parameter numbers: 2
```

</pre>
</td>
</tr>
</table>

### 2. Model Verification

The implementation of the model algorithm is irrelevant to the framework. The trained parameters can be converted into the [checkpoint](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html) file of MindSpore and loaded to the network for inference verification.

For details about the model verification process, see [ResNet Network Migration](https://www.mindspore.cn/docs/en/master/migration_guide/sample_code.html#model-validation).

### 3. Inference Verification

After confirming that the model structures are the same, you are advised to perform inference verification again. In addition to models, the entire inference process also involves datasets and metrics. When the inference results are inconsistent, you can use the control variable method to gradually rectify the fault.

For details about the inference verification process, see [ResNet Network Migration](https://www.mindspore.cn/docs/en/master/migration_guide/sample_code.html#inference-process).

### 4. Training Accuracy

After the inference verification is complete, the basic model, data processing, and metrics calculation are normal. If the training accuracy is still abnormal, how do we locate the fault?

- Add loss scale. On Ascend, operators such as Conv, Sort, and TopK can only be float16. MatMul is recommended to be float16 due to performance problems. Therefore, it is recommended that loss scale be used as a standard configuration for network training.

  The list of operators only supports float16 on Ascend:

  | type | operators |
  | ------  | ------ |
  | Pool    | AdaptiveMaxPool2D, AvgPool3D, AvgPool, MaxPool, MaxPoolWithArgmax, Pooling |
  | RNN     | LSTM, DynamicRNN, GRUV2 |
  | Conv    | Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, DepthwiseConv2dNative |
  | Matmul (float32 is too slow and needs to be cast to float16) | MatMul, BatchMatMul |
  | Sort | Sort, TopK |
  | Others | BoundingBoxEncode, ExtractImagePatches, ExtractVolumePatches, FusedDbnDw, IOU, NewIm2Col, NMSWithMask |

  ```python
  import mindspore as ms
  from mindspore import nn
  from mindspore.train import Model
  # Model
  loss_scale_manager = ms.amp.FixedLossScaleManager(drop_overflow_update=False) # Static loss scale
  # loss_scale_manager = ms.amp.DynamicLossScaleManager()   # Dynamic loss scale

  # 1. General process
  loss = nn.MSELoss()
  opt = nn.Adam(params=msnet.trainable_params(), learning_rate=0.01)
  model = Model(network=msnet, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager)

  # 2. Self-packaged forward network and loss function
  msnet.to_float(ms.float16)
  loss.to_float(ms.float32)
  net_with_loss = nn.WithLossCell(msnet, loss)
  # It is recommended that loss_fn be used for the mixed precision of the model. Otherwise, float16 is used for calculation of the loss part, which may cause overflow.
  model = Model(network=net_with_loss, optimizer=opt)

  # 3. Self-packaged training process
  scale_sense = nn.FixedLossScaleUpdateCell(1)#(config.loss_scale) # Static loss scale
  # scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=config.loss_scale,
  #                                             scale_factor=2, scale_window=1000) # Dynamic loss scale
  train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer=opt, scale_sense=scale_sense)
  model = Model(network=train_net)
  ```

- Check whether overflow occurs. When loss scale is added, overflow detection is added by default to monitor the overflow result. If overflow occurs continuously, you are advised to use the [dump data](https://mindspore.cn/tutorials/experts/en/master/debug/dump.html) of MindSpore Insight to check why overflow occurs.

  ```python
  import numpy as np
  from mindspore import dataset as ds

  def get_data(num, w=2.0, b=3.0):
      for _ in range(num):
          x = np.random.uniform(-10.0, 10.0)
          noise = np.random.normal(0, 1)
          y = x * w + b + noise
          yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

  def create_dataset(num_data, batch_size=16, repeat_size=1):
      input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
      input_data = input_data.batch(batch_size, drop_remainder=True)
      input_data = input_data.repeat(repeat_size)
      return input_data

  train_net.set_train()
  dataset = create_dataset(1600)
  iterator = dataset.create_tuple_iterator()
  for i, data in enumerate(iterator):
      loss, overflow, scaling_sens = train_net(*data)
      print("step: {}, loss: {}, overflow:{}, scale:{}".format(i, loss, overflow, scaling_sens))
  ```

  ```text
  step: 0, loss: 138.42825, overflow:False, scale:1.0
  step: 1, loss: 118.172104, overflow:False, scale:1.0
  step: 2, loss: 159.14542, overflow:False, scale:1.0
  step: 3, loss: 150.65671, overflow:False, scale:1.0
  ... ...
  step: 97, loss: 69.513245, overflow:False, scale:1.0
  step: 98, loss: 51.903114, overflow:False, scale:1.0
  step: 99, loss: 42.250656, overflow:False, scale:1.0
  ```

- Check the optimizer, loss, and parameter initialization. In addition to the model and dataset, only the optimizer, loss, and parameter initialization are added in the entire training process. If the training is abnormal, check the optimizer, loss, and parameter initialization. Especially for loss and parameter initialization, there is a high probability that the problem occurs.
- Check whether to add seeds for multiple devices to ensure that the initialization of multiple SIM cards is consistent. Determine whether to perform gradient aggregation during [customized training](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/training_and_evaluation.html#training-process).

  ```python
  import mindspore as ms
  ms.set_seed(1) # The random seeds of MindSpore, NumPy, and dataset are fixed. The random seed of the API needs to be set in the API attribute.
  ```

- Check whether the data processing meets the expectation through visualization. Focus on data shuffle and check whether data mismatch occurs.

For details about more accuracy debugging policies, see [Accuracy Debugging](https://mindspore.cn/mindinsight/docs/en/master/accuracy_problem_preliminary_location.html).
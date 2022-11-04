# 基于Fourier Neural Operator的Navier-Stokes equation求解

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindflow/docs/source_zh_cn/data_driven/fno2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

计算流体力学是21世纪流体力学领域的重要技术之一，其通过使用数值方法在计算机中对流体力学的控制方程进行求解，从而实现流动的分析、预测和控制。传统的有限元法（finite element method，FEM）和有限差分法（finite difference method，FDM）常囿于复杂的仿真流程（物理建模，网格划分，数值离散，迭代求解等）和较高的计算成本，往往效率低下。因此，借助AI提升流体仿真效率是十分必要的。

近年来，随着神经网络的迅猛发展，为科学计算提供了新的范式。经典的神经网络是在有限维度的空间进行映射，只能学习与特定离散化相关的解。与经典神经网络不同，傅里叶神经算子（Fourier Neural Operator，FNO）是一种能够学习无限维函数空间映射的新型深度学习架构。该架构可直接学习从任意函数参数到解的映射，用于解决一类偏微分方程的求解问题，具有更强的泛化能力。更多信息可参考[原文](https://arxiv.org/abs/2010.08895)。

本案例教程介绍利用傅里叶神经算子的纳维-斯托克斯方程（Navier-Stokes equation）求解方法。

## 纳维-斯托克斯方程（Navier-Stokes equation）

纳维-斯托克斯方程（Navier-Stokes equation）是计算流体力学领域的经典方程，是一组描述流体动量守恒的偏微分方程，简称N-S方程。它在二维不可压缩流动中的涡度形式如下：

$$
\partial_t w(x, t)+u(x, t) \cdot \nabla w(x, t)=\nu \Delta w(x, t)+f(x), \quad x \in(0,1)^2, t \in(0, T]
$$

$$
\nabla \cdot u(x, t)=0, \quad x \in(0,1)^2, t \in[0, T]
$$

$$
w(x, 0)=w_0(x), \quad x \in(0,1)^2
$$

其中$u$表示速度场，$w=\nabla \times u$表示涡度，$w_0(x)$表示初始条件，$\nu$表示粘度系数，$f(x)$为外力合力项。

## 问题描述

本案例利用Fourier Neural Operator学习某一个时刻对应涡度到下一时刻涡度的映射，实现二维不可压缩N-S方程的求解：

$$
w_t \mapsto w(\cdot, t+1)
$$

MindFlow求解该问题的具体流程如下：

1. 配置网络与训练参数。
2. 创建训练数据集。
3. 构建神经网络。
4. 定义损失函数。
5. 定义模型测试模块。
6. 模型训练。

## Fourier Neural Operator

Fourier Neural Operator模型构架如下图所示。图中$w_0(x)$表示初始涡度，通过Lifting Layer实现输入向量的高维映射，然后将映射结果作为Fourier Layer的输入，进行频域信息的非线性变换，最后由Decoder Layer将变换结果映射至最终的预测结果$w_1(x)$。

Lifting Layer、Fourier Layer以及Decoder Layer共同组成了Fourier Neural Operator。

![Fourier Neural Operator模型构架](images/FNO.png)

Fourier Layer网络结构如下图所示。图中V表示输入向量，上框表示向量经过傅里叶变换后，经过线性变换R，过滤高频信息，然后进行傅里叶逆变换；另一分支经过线性变换W，最后通过激活函数，得到Fourier Layer输出向量。

![Fourier Layer网络结构](images/FNO-2.png)

### 导入依赖

导入本教程所依赖模块与接口：

```python
import os
import argparse
import datetime
import numpy as np

import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore import Tensor, context
from mindspore.train import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.train import DynamicLossScaleManager

from mindflow.cell.neural_operators import FNO2D
from mindflow.solver import Solver

from src.callback import PredictCallback
from src.lr_scheduler import warmup_cosine_annealing_lr
from src.dataset import create_dataset
from src.utils import load_config
from src.loss import RelativeRMSELoss
```

### 创建数据集

本案例根据Zongyi Li在 [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/pdf/2010.08895.pdf) 一文中对数据集的设置生成训练数据集与测试数据集。具体设置如下：

基于周期性边界，生成满足如下分布的初始条件$w_0(x)$：

$$
w_0 \sim \mu, \mu=\mathcal{N}\left(0,7^{3 / 2}(-\Delta+49 I)^{-2.5}\right)
$$

外力项设置为：

$$
f(x)=0.1\left(\sin \left(2 \pi\left(x_1+x_2\right)\right)+\right.\cos(2 \pi(x_1+x_2)))
$$

采用`Crank-Nicolson`方法生成数据，时间步长设置为1e-4，最终数据以每 t = 1 个时间单位记录解。所有数据均在256×256的网格上生成，并被下采样至64×64网格。本案例选取粘度系数$\nu=1e−5$，训练集样本量为19000个，测试集样本量为3800个。

```python
# create dataset for train
config = load_config('path/to/config')
data_params = config["data"]
model_params = config["model"]
train_dataset = create_dataset(data_params,
                               input_resolution=model_params["input_resolution"],
                               shuffle=True)
test_input = np.load(os.path.join(data_params["path"], "test/inputs.npy"))
test_label = np.load(os.path.join(data_params["path"], "test/label.npy"))
```

代码中`load_config`引用自`utils.py`，`create_dataset`引用自`dataset.py`，data_params与model_params中对应参数分别在配置文件中配置。

### 构建神经网络

网络由1层Lifting layer、多层Fourier Layer以及1层Docoder layer叠加组成：

- Lifting layer对应样例代码中`FNO2D.fc0`，将输出数据$x$映射至高维；

- 多层Fourier Layer的叠加对应样例代码中`FNO2D.fno_seq`，本案例采用离散傅里叶变换实现时域与频域的转换；

- Docoder layer对应代码中`FNO2D.fc1`与`FNO2D.fc2`，获得最终的预测值。

```python
class FNOBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, modes1, resolution=211, gelu=True, compute_dtype=mstype.float16):
        super().__init__()
        self.conv = SpectralConv2dDft(in_channels, out_channels, modes1, resolution, compute_dtype=compute_dtype)
        self.w = nn.Conv2d(in_channels, out_channels, 1, weight_init='HeUniform').to_float(compute_dtype)

        if gelu:
            self.act = ops.GeLU()
        else:
            self.act = ops.Identity()

    def construct(self, x):
        return self.act(self.conv(x) + self.w(x))

class FNO2D(nn.Cell):
    def __init__(self,
                 input_dims,
                 output_dims,
                 resolution,
                 modes,
                 width=20,
                 depth=4,
                 mlp_ratio=4,
                 compute_dtype=mstype.float32):
        super().__init__()
        check_param_type(input_dims, "input_dims", data_type=int, exclude_type=bool)
        check_param_type(output_dims, "output_dims", data_type=int, exclude_type=bool)
        check_param_type(resolution, "resolution", data_type=int, exclude_type=bool)
        check_param_type(modes, "modes", data_type=int, exclude_type=bool)
        if modes < 1:
            raise ValueError("modes must at least 1, but got mode: {}".format(modes))

        self.modes1 = modes
        self.channels = width
        self.fc_channel = mlp_ratio * width
        self.fc0 = nn.Dense(input_dims + 2, self.channels, has_bias=False).to_float(compute_dtype)
        self.layers = depth

        self.fno_seq = nn.SequentialCell()
        for _ in range(self.layers - 1):
            self.fno_seq.append(FNOBlock(self.channels, self.channels, modes1=self.modes1, resolution=resolution,
                                         compute_dtype=compute_dtype))
        self.fno_seq.append(
            FNOBlock(self.channels, self.channels, self.modes1, resolution=resolution, gelu=False,
                     compute_dtype=compute_dtype))

        self.fc1 = nn.Dense(self.channels, self.fc_channel, has_bias=False).to_float(compute_dtype)
        self.fc2 = nn.Dense(self.fc_channel, output_dims, has_bias=False).to_float(compute_dtype)

        self.grid = Tensor(get_grid_2d(resolution), dtype=mstype.float32)
        self.concat = ops.Concat(axis=-1)
        self.act = ops.GeLU()

    def construct(self, x: Tensor):
        batch_size = x.shape[0]

        grid = self.grid.repeat(batch_size, axis=0)
        x = P.Concat(-1)((x, grid))
        x = self.fc0(x)
        x = P.Transpose()(x, (0, 3, 1, 2))

        x = self.fno_seq(x)

        x = P.Transpose()(x, (0, 2, 3, 1))
        x = self.fc1(x)
        x = self.act(x)
        output = self.fc2(x)

        return output
```

基于上述网络结构，进行模型初始化，其中model_params中的配置可在配置文件中修改。

```python
model = FNO2D(input_dims=model_params["input_dims"],
              output_dims=model_params["output_dims"],
              resolution=model_params["input_resolution"],
              modes=model_params["modes"],
              width=model_params["width"],
              depth=model_params["depth"]
             )
```

### 定义损失函数

使用相对均方根误差作为网络训练损失函数：

```python
import mindspore
import mindspore.nn as nn
from mindspore import ops

class RelativeRMSELoss(nn.LossBase):
    def __init__(self, reduction="sum"):
        super(RelativeRMSELoss, self).__init__(reduction=reduction)

    def construct(self, prediction, label):
        prediction = ops.Cast()(prediction, mindspore.float32)
        batch_size = ops.shape[0]
        diff_norms = ops.square(prediction.reshape(batch_size, -1) - label.reshape(batch_size, -1)).sum(axis=1)
        label_norms = ops.square(label.reshape(batch_size, -1)).sum(axis=1)
        rel_error = ops.div(ops.sqrt(diff_norms), ops.sqrt(label_norms))
        return self.get_loss(rel_error)
```

### 定义模型测试模块

通过自定义的PredictCallback函数，实现边训练边推理的功能。用户可以直接加载测试数据集，每训练n个epoch后输出一次测试集上的推理精度，n的大小通过配置文件中的eval_interval进行设置。

```python
class PredictCallback(Callback):
    def __init__(self,
                 model,
                 inputs,
                 label,
                 config,
                 summary_dir):
        super(PredictCallback, self).__init__()
        self.model = model
        self.inputs = inputs
        self.label = label
        self.length = label.shape[0]
        self.summary_dir = summary_dir
        self.predict_interval = config.get("eval_interval", 3)
        self.batch_size = config.get("test_batch_size", 1)
        self.rel_rmse_error = 1.0
        self.T = 10
        print("check test dataset shape: {}, {}".format(self.inputs.shape, self.label.shape))

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            print("================================Start Evaluation================================")
            time_beg = time.time()
            rel_rmse_error = 0.0
            max_error = 0.0
            for i in range(self.length):
                for j in range(self.T - 1, self.T + 9):
                    label = self.label[i:i + 1, j]
                    if j == self.T - 1:
                        test_batch = Tensor(self.inputs[i:i + 1, j], dtype=mstype.float32)
                    else:
                        test_batch = Tensor(prediction)
                    prediction = self.model(test_batch)
                    prediction = prediction.asnumpy()
                    rel_rmse_error_step = self._calculate_error(label, prediction)
                    rel_rmse_error += rel_rmse_error_step

                    if rel_rmse_error_step >= max_error:
                        max_error = rel_rmse_error_step

            self.rel_rmse_error = rel_rmse_error / (self.length * 10)
            print("mean rel_rmse_error:", self.rel_rmse_error)
            self.summary_record.add_value('scalar', 'rel_rmse_error', Tensor(self.rel_rmse_error))
            print("=================================End Evaluation=================================")
            print("predict total time: {} s".format(time.time() - time_beg))
            self.summary_record.record(cb_params.cur_step_num)

    def _calculate_error(self, label, prediction):
        """calculate l2-error to evaluate accuracy"""
        rel_error = np.sqrt(np.sum(np.square(label.reshape(self.batch_size, -1) -
                                             prediction.reshape(self.batch_size, -1)))) / \
                    np.sqrt(np.sum(np.square(prediction.reshape(self.batch_size, -1))))
        return rel_error

    def get_rel_rmse_error(self):
        return self.rel_rmse_error
```

PredictCallback初始化：

```python
pred_cb = PredictCallback(model=model,
                          inputs=test_input,
                          label=test_label,
                          config=callback_params,
                          summary_dir=summary_dir)
```

### 模型训练与推理

Solver类是模型训练和推理的接口。输入优化器、网络模型、损失函数、损失缩放策略等，即可定义求解器对象solver。在该案例中利用MindSpore + Ascend混合精度模式训练网络，从而完成2维N-S方程求解。代码中optimizer_params、model_params对应各项参数均在配置文件中修改。

```python
# optimizer
steps_per_epoch = train_dataset.get_dataset_size()
lr = warmup_cosine_annealing_lr(lr=optimizer_params["initial_lr"],
                                steps_per_epoch=steps_per_epoch,
                                warmup_epochs=optimizer_params["warmup_epochs"],
                                max_epoch=optimizer_params["train_epochs"])

optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))

# prepare loss function
loss_scale = DynamicLossScaleManager()
loss_fn = RelativeRMSELoss()

# define solver
solver = Solver(model,
                optimizer=optimizer,
                loss_scale_manager=loss_scale,
                loss_fn=loss_fn,
                )
solver.train(epoch=optimizer_params["train_epochs"],
             train_dataset=train_dataset,
             callbacks=[LossMonitor(), TimeMonitor(), pred_cb],
             dataset_sink_mode=True)
```

## 网络训练结果

运行结果如下，训练50个迭代轮次后，网络损失函数值降至1.475，在测试集上的相对均方根误差为0.110。

```python
......
epoch: 41 step: 1000, loss is 1.417490005493164
Train epoch time: 6512.500 ms, per step time: 6.513 ms
epoch: 42 step: 1000, loss is 1.6001394987106323
Train epoch time: 6516.459 ms, per step time: 6.516 ms
epoch: 43 step: 1000, loss is 1.64013671875
Train epoch time: 6520.781 ms, per step time: 6.521 ms
epoch: 44 step: 1000, loss is 1.7954413890838623
Train epoch time: 6520.548 ms, per step time: 6.521 ms
epoch: 45 step: 1000, loss is 1.639083743095398
Train epoch time: 6519.727 ms, per step time: 6.520 ms
epoch: 46 step: 1000, loss is 2.7023866176605225
Train epoch time: 6513.133 ms, per step time: 6.513 ms
epoch: 47 step: 1000, loss is 1.5318703651428223
Train epoch time: 6509.813 ms, per step time: 6.510 ms
epoch: 48 step: 1000, loss is 2.2350616455078125
Train epoch time: 6522.118 ms, per step time: 6.522 ms
epoch: 49 step: 1000, loss is 2.0657312870025635
Train epoch time: 6514.847 ms, per step time: 6.515 ms
epoch: 50 step: 1000, loss is 1.4754825830459595
Train epoch time: 6577.887 ms, per step time: 6.578 ms
================================Start Evaluation================================
mean rel_rmse_error: 0.11016936695948243
=================================End Evaluation=================================
......
```

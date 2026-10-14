# MindSpore Lite支持的硬件后端列表

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.11.0/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/r2.11.0/docs/lite/docs/source_zh_cn/reference/operator_list_lite.md)

## Abs

### 功能

逐元素计算绝对值。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## AbsGrad

### 功能

计算绝对值函数的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Activation

### 功能

激活函数。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## ActivationGrad

### 功能

计算特定激活函数的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Adam

### 功能

执行Adam优化器的一次参数更新步骤。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## AddFusion

### 功能

逐元素计算加法。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32、Int8
- Ascend：FP16

</br></br></br>

---

## AdderFusion

### 功能

基于加法的卷积运算。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## AddGrad

### 功能

计算加法操作的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## AddN

### 功能

对N个相同形状和数据类型的输入张量进行逐元素相加。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Affine

### 功能

对输入张量执行仿射变换。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## All

### 功能

判断张量中所有元素在指定维度上是否都为True（非零）。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## AllGather

### 功能

分布式集合通信操作。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## ApplyMomentum

### 功能

执行带动量的随机梯度下降的一次参数更新步骤。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## ArgmaxFusion

### 功能

求某一维度最大值。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## ArgminFusion

### 功能

求某一维度最小值。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Assert

### 功能

断言。

### 支持的硬件后端

- CPU：FP16、FP32、Bool

</br></br></br>

---

## Assign

### 功能

将一个值赋值给一个变量。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## AvgPoolFusion

### 功能

平均池化。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## AvgPoolGrad

### 功能

计算平均池化层的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## BatchNorm

### 功能

批量归一化。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## BatchNormGrad

### 功能

计算批量归一化层的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## BatchToSpace

### 功能

空间到批次变换的逆操作。

### 支持的硬件后端

- CPU：FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## BatchToSpaceND

### 功能

BatchToSpace的ND通用版本。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## BiasAdd

### 功能

将偏置向量添加到输入张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## BiasAddGrad

### 功能

计算BiasAdd操作的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## BinaryCrossEntropy

### 功能

计算二元交叉熵损失。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## BinaryCrossEntropyGrad

### 功能

计算二元交叉熵损失函数的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## BroadcastTo

### 功能

按照 NumPy 广播规则，把一个张量沿指定的维度“扩展/复制”成更大的形状。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool

</br></br></br>

---

## Call

### 功能

调用一个子计算图或函数。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool
- Ascend：FP16

</br></br></br>

---

## Cast

### 功能

数据类型转换。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Ceil

### 功能

向上取整。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Clip

### 功能

限制元素范围。

### 支持的硬件后端

- CPU：FP32、Int32
- Ascend：FP16

</br></br></br>

---

## Concat

### 功能

拼接张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## ConstantOfShape

### 功能

生成一个与输入形状相同的张量，并用指定常量填充。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## Conv2DFusion

### 功能

2D卷积。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Conv2DBackpropFilterFusion

### 功能

计算普通卷积操作对卷积核的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Conv2DBackpropInputFusion

### 功能

计算普通卷积操作对输入数据的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Conv2dTransposeFusion

### 功能

执行转置卷积运算。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Cos

### 功能

逐元素计算余弦。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Crop

### 功能

从输入图像或特征图中裁剪出一个指定区域。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8

</br></br></br>

---

## CropAndResize

### 功能

从输入图像中根据一组边界框裁剪出区域，然后将每个区域缩放到统一大小。

### 支持的硬件后端

- CPU：FP32
- Kirin NPU：FP16

</br></br></br>

---

## CumSum

### 功能

累计元素和。

### 支持的硬件后端

- CPU：FP32、Int32
- Ascend：FP16

</br></br></br>

---

## CustomExtractFeatures

### 功能

自定义特征提取算子。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## CustomNormalize

### 功能

自定义归一化算子。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## CustomPredict

### 功能

自定义预测算子。

### 支持的硬件后端

- CPU：FP32、Int32

</br></br></br>

---

## DEConv2DGradFilter

### 功能

计算转置卷积对卷积核的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## DepthToSpace

### 功能

将深度数据重新排列到空间维度中。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## DetectionPostProcess

### 功能

目标检测后处理。

### 支持的硬件后端

- CPU：FP32、Int8、UInt8

</br></br></br>

---

## DivFusion

### 功能

逐元素除法。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## DivGrad

### 功能

计算除法操作的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Dropout

### 功能

随机将输入张量的部分元素置 0。

### 支持的硬件后端

- CPU：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## DropoutGrad

### 功能

计算Dropout操作的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## DynamicQuant

### 功能

动态将浮点张量量化为uint8类型。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Eltwise

### 功能

元素级运算。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Elu

### 功能

激活函数，对负输入使用指数修正。

### 支持的硬件后端

- CPU：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Equal

### 功能

判断输入是否相等。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## EmbeddingLookupFusion

### 功能

优化版的词嵌入查找，将整数索引映射为密集向量。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Erf

### 功能

误差函数。

### 支持的硬件后端

- CPU：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## ExpFusion

### 功能

逐元素取指数。

### 支持的硬件后端

- CPU：FP16、FP32
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## ExpandDims

### 功能

在指定位置插入长度为1的维度。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## Fill

### 功能

生成一个填充指定常量的张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Flatten

### 功能

数据按维度展开。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## FlattenGrad

### 功能

计算Flatten操作的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Floor

### 功能

向下取整。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## FloorDiv

### 功能

逐元素向下取整除法。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## FloorMod

### 功能

逐元素取模运算，结果的符号与除数一致。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## FullConnection

### 功能

全连接层。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## FusedBatchNorm

### 功能

对输入做标准化。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- Ascend：FP16

</br></br></br>

---

## Gather

### 功能

沿单一维度收集指定索引位置的元素。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## GatherD

### 功能

根据索引张量收集输入张量中的元素。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool
- Ascend：FP16

</br></br></br>

---

## GatherNd

### 功能

根据索引张量从输入张量中收集指定位置的元素。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## GLU

### 功能

门控线性单元激活函数，将输入拆分为两部分并逐元素相乘。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Greater

### 功能

逐元素比较两个张量，返回A>B的逻辑结果（True/False）。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## GreaterEqual

### 功能

逐元素比较两个张量，返回 A≥B的逻辑结果（True/False）。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## GroupNormFusion

### 功能

融合优化的组归一化。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## GRU

### 功能

门控循环单元，简化版LSTM。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## HashtableLookup

### 功能

哈希表查找。

### 支持的硬件后端

- CPU：FP32、Int32

</br></br></br>

---

## InstanceNorm

### 功能

实例归一化。

### 支持的硬件后端

- CPU：FP16、FP32
- Kirin NPU：FP16
- Ascend：FP16

</br></br></br>

---

## InvertPermutation

### 功能

反转置换索引。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## IsFinite

### 功能

检测张量中每个元素是否为有限值（非inf/NaN）。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## L2NormalizeFusion

### 功能

融合优化的L2归一化。

### 支持的硬件后端

- CPU：FP32、Int8、UInt8

</br></br></br>

---

## LayerNormFusion

### 功能

融合优化的层归一化。

### 支持的硬件后端

- CPU：FP16、FP32、Int8
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## LayerNormGrad

### 功能

计算层归一化的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## LeakyReLU

### 功能

带泄漏的ReLU激活函数，对负输入给予微小斜率。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Less

### 功能

逐元素比较两个张量，返回 A<B的逻辑结果。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## LessEqual

### 功能

逐元素比较A ≤ B，返回布尔张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## LRN

### 功能

局部响应归一化。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## Log

### 功能

逐元素求对数。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Log1p

### 功能

计算log(1+X)。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## LogGrad

### 功能

计算对数函数的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## LogicalAnd

### 功能

逐元素逻辑与运算。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## LogicalNot

### 功能

逐元素逻辑非运算。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## LogicalOr

### 功能

逐元素逻辑或运算。

### 支持的硬件后端

- CPU：FP16、FP32、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## LogSoftmax

### 功能

对输入向量进行softmax操作，然后再对softmax结果取对数。

### 支持的硬件后端

- CPU：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## LshProjection

### 功能

局部敏感哈希投影。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## LSTM

### 功能

长短期记忆网络单元。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## LSTMGrad

### 功能

计算LSTM对隐状态的反向传播梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## LSTMGradData

### 功能

计算LSTM对输入数据的反向传播梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## LSTMGradWeight

### 功能

计算LSTM对权重的反向传播梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## MatMulFusion

### 功能

对2个输入做矩阵乘法运算。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Maximum

### 功能

取元素级最大值。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## MaximumGrad

### 功能

计算最大值函数的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## MaxPoolFusion

### 功能

最大池化。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## MaxPoolGrad

### 功能

计算最大池化层的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Merge

### 功能

从一组输入张量中获取第一个可用的张量并输出。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Minimum

### 功能

取元素级最小值。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## MinimumGrad

### 功能

计算最小值函数的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Mod

### 功能

返回除法元素的余数。

### 支持的硬件后端

- CPU：FP32、Int32
- Ascend：FP16

</br></br></br>

---

## MulFusion

### 功能

逐元素乘法。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## MulGrad

### 功能

计算乘法操作的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Neg

### 功能

逐元素求负数。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## NegGrad

### 功能

计算取负操作的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## NLLLoss

### 功能

计算负对数似然损失。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## NLLLossGrad

### 功能

计算NLLLoss的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## NotEqual

### 功能

逐元素比较两个张量，返回 A != B的逻辑结果。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## NonMaxSuppression

### 功能

非极大值抑制。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## NonZero

### 功能

返回输入张量中所有非零元素的索引。

### 支持的硬件后端

- CPU：Bool
- Ascend：FP16

</br></br></br>

---

## OneHot

### 功能

将整数索引张量转换为独热编码表示。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- GPU (Mali/Adreno)：FP16、FP32、Int32

</br></br></br>

---

## OnesLike

### 功能

创建一个与输入张量 X形状完全相同但所有元素值均为1的新张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## PadFusion

### 功能

将输入张量加上指定的padding，使其达到指定的大小。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## PartialFusion

### 功能

部分融合。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool

</br></br></br>

---

## PowFusion

### 功能

逐元素求幂。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## PowerGrad

### 功能

计算幂运算的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## PriorBox

### 功能

生成先验框。

### 支持的硬件后端

- CPU：FP32、Int8、UInt8
- Ascend：FP16

</br></br></br>

---

## PReLUFusion

### 功能

PRelu激活函数。

### 支持的硬件后端

- CPU：FP16、FP32
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## QuantDTypeCast

### 功能

执行量化数据类型转换。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8

</br></br></br>

---

## RaggedRange

### 功能

生成非均匀间隔的序列。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## RandomNormal

### 功能

生成一个张量，其中的值从正态分布中随机采样。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## RandomStandardNormal

### 功能

生成服从标准正态分布的随机数张量。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Range

### 功能

生成某个区间内的元素。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## Rank

### 功能

返回输入张量的维度数。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## RealDiv

### 功能

逐元素除法。

### 支持的硬件后端

- CPU：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Reciprocal

### 功能

返回倒数。

### 支持的硬件后端

- CPU：FP16、FP32、Int8
- Kirin NPU：FP16
- Ascend：FP16

</br></br></br>

---

## ReduceFusion

### 功能

归约操作。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## ReduceScatter

### 功能

分布式操作，将输入张量分段后分发到各设备，每设备仅保留一段结果。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Reshape

### 功能

改变张量形状，总元素个数不变。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## Resize

### 功能

对输入张量进行上采样或调整大小。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## ResizeGrad

### 功能

计算Resize的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## ReverseV2

### 功能

沿指定轴反转张量。

### 支持的硬件后端

- CPU：FP32、Int32

</br></br></br>

---

## ReverseSequence

### 功能

对输入张量的可变长度序列进行部分反转。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## ROIPooling

### 功能

区域兴趣池化。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## Round

### 功能

四舍五入到最接近的整数数值。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Rsqrt

### 功能

逐元素计算平方根倒数，用于归一化。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## RsqrtGrad

### 功能

计算平方根倒数的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Select

### 功能

根据条件从两个张量中选择元素。

### 支持的硬件后端

- CPU：FP32、Bool

</br></br></br>

---

## Selu

### 功能

自归一化指数线性单元激活函数。

### 支持的硬件后端

无

</br></br></br>

---

## ScaleFusion

### 功能

将缩放操作与相邻算子融合。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## ScatterNd

### 功能

根据索引将更新张量中的值散射到输出张量的指定位置。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## ScatterNdUpdate

### 功能

使用给定值以及输入索引更新输入数据的值。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## SGD

### 功能

随机梯度下降优化器。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## Shape

### 功能

获得张量shape。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SigmoidCrossEntropyWithLogits

### 功能

结合Sigmoid激活和交叉熵损失。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## SigmoidCrossEntropyWithLogitsGrad

### 功能

计算带Sigmoid的交叉熵损失的梯度。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## Sin

### 功能

逐元素计算正弦。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## Size

### 功能

获取张量维度大小。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- Ascend：FP16

</br></br></br>

---

## SliceFusion

### 功能

张量切片操作。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SkipGram

### 功能

Skip-gram模型的核心操作，用于词向量训练。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## SmoothL1Loss

### 功能

平滑L1损失。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## SmoothL1LossGrad

### 功能

计算平滑L1损失的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Softmax

### 功能

归一化操作。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SoftmaxGrad

### 功能

计算Softmax的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Softplus

### 功能

平滑的ReLU变体。

### 支持的硬件后端

- CPU：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SpaceToBatch

### 功能

高度和宽度维度的值移至深度维度。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SpaceToBatchND

### 功能

将空间维度的数据块拆分到批次维度。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## SpaceToDepth

### 功能

将空间数据重组为深度通道。

### 支持的硬件后端

- CPU：FP16、FP32
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## SparseToDense

### 功能

将稀疏表示转换为密集张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- GPU (Mali/Adreno)：FP16、FP32、Int32

</br></br></br>

---

## SparseSoftmaxCrossEntropyWithLogits

### 功能

稀疏标签的Softmax交叉熵。

### 支持的硬件后端

- CPU：FP32
- Ascend：FP16

</br></br></br>

---

## Splice

### 功能

沿指定轴连接输入张量的多个切片或范围。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Split

### 功能

将输入张量沿指定轴分割成多个较小的输出张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SplitWithOverlap

### 功能

带重叠的分割张量。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Sqrt

### 功能

逐元素开根号。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SqrtGrad

### 功能

计算平方根的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Square

### 功能

逐元素平方。

### 支持的硬件后端

- CPU：FP16、FP32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SquaredDifference

### 功能

逐元素计算 (A-B)²。

### 支持的硬件后端

- CPU：FP16、FP32
- GPU (Mali/Adreno)：FP16、FP32

</br></br></br>

---

## Squeeze

### 功能

移除维度为1的维度。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- GPU (Mali/Adreno)：FP16、FP32、Int32

</br></br></br>

---

## StridedSlice

### 功能

Tensor切片。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## StridedSliceGrad

### 功能

计算切片操作的梯度。

### 支持的硬件后端

- CPU：FP16、FP32

</br></br></br>

---

## Stack

### 功能

沿新轴堆叠多个张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SubFusion

### 功能

逐元素相减。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## SubGrad

### 功能

计算减法的梯度。

### 支持的硬件后端

- CPU：FP32

</br></br></br>

---

## Switch

### 功能

根据布尔条件选择输出分支。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool

</br></br></br>

---

## SwitchLayer

### 功能

在模型中选择执行不同的子网络分支。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool

</br></br></br>

---

## TensorListFromTensor

### 功能

将普通张量转换为张量列表，按指定轴分割。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## TensorListGetItem

### 功能

从张量列表中获取指定索引位置的张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## TensorListReserve

### 功能

预分配一个空张量列表，指定元素数据类型和初始容量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## TensorListSetItem

### 功能

将张量插入张量列表的指定位置。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## TensorListStack

### 功能

将张量列表堆叠为一个普通张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## TensorScatterAdd

### 功能

根据索引将更新张量的值分散添加到目标张量的指定位置。

### 支持的硬件后端

- CPU：FP32、Int32

</br></br></br>

---

## TileFusion

### 功能

平铺给定矩阵。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool
- Kirin NPU：FP16
- Ascend：FP16

</br></br></br>

---

## TopKFusion

### 功能

从输入张量中返回topK个元素。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8
- Ascend：FP16

</br></br></br>

---

## Transpose

### 功能

张量转置。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32
- Ascend：FP16

</br></br></br>

---

## UniformReal

### 功能

生成服从均匀分布的随机数张量。

### 支持的硬件后端

- CPU：FP32、Int32

</br></br></br>

---

## Unique

### 功能

返回输入张量中的唯一值，并可返回该值的索引和计数。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## UnsortedSegmentSum

### 功能

对张量进行分段求和，不要求分段索引有序。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## Unsqueeze

### 功能

将输入张量添加一个新的维度。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Int8、UInt8、Bool
- Kirin NPU：FP16
- GPU (Mali/Adreno)：FP16、FP32、Int32

</br></br></br>

---

## Unstack

### 功能

沿指定轴拆分张量为多个子张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

</br></br></br>

---

## Where

### 功能

元素选择。

### 支持的硬件后端

- CPU：FP16、FP32、Int32、Bool

</br></br></br>

---

## ZerosLike

### 功能

生成与输入张量形状相同但全为0的新张量。

### 支持的硬件后端

- CPU：FP16、FP32、Int32

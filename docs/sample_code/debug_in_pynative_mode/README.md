# 文件说明

> 测试文件运行环境为: mindspore-ascend 1.0.1。

- `01_single_op.py`：执行单个算子，并打印相关结果，示例输出`([[[[-0.02190447 -0.05208071 -0.0……5208071 -0.05208071 -0.06265172] ... [ 0.05016355 0.03958241 0.03958241 0.03958241 0.03443141]]]])`。

- `02_single_function.py`：若干算子组合成一个函数，并打印相关结果，示例输出`([[3. 3. 3.] [3. 3. 3.] [3. 3. 3.]])`。

- `03_staging.py`：MindSpore提供Staging功能，该功能可以在PyNative模式下将Python函数或者Python类的方法编译成计算图，通过图优化等技术提高运行速度，示例输出`([[3. 3. 3. 3.] [3. 3. 3. 3.] [3. 3. 3. 3.] [3. 3. 3. 3.]])`。

- `04_staging_2.py`：加装了`ms_function`装饰器的函数中，如果包含不需要进行参数训练的算子（如`pooling`、`tensor_add`等算子），则这些算子可以在被装饰的函数中直接调用，示例输出`[[2. 2. 2. 2.] [2. 2. 2. 2.] [2. 2. 2. 2.] [2. 2. 2. 2.]]`。

- `05_staging_3.py`：被装饰的函数中包含了需要进行参数训练的算子（如`Convolution`、`BatchNorm`等算子），则这些算子必须在被装饰的函数之外完成实例化操作，示例输出`[[[[ 0.10377571 -0.0182163 -0.05221086] ... [ 0.0377498 -0.06117418 0.00546303]]]]`。

- `06_grad.py`：PyNative模式下，还可以支持单独求梯度的操作，示例输出`(Tensor(shape=[], dtype=Int32, value=2), Tensor(shape=[], dtype=Int32, value=1))`。

- `07_lenet.py`：LeNet示例输出：2.3050091。
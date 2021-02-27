# tensor_dot功能差异

PyTorch: 计算两个相同shape的tensor的点乘（内积），仅支持1D。

MindSpore：计算两个tensor在任意轴上的点乘，支持任意维度的tensor，但指定的轴对应的shape要相等。当输入为1D，轴设定为0时和PyTorch的功能一致。

# Function Differences of tensor_dot

PyTorch: Calculate the dot product(inner product) of two tensors of the same shape, only support 1D.

MindSpore：Calculate the dot product of two tensors on any axis. Support tensor of any dimension, but the shape corresponding to the specified axis should be equal. The function of the PyTorch is the same when the input is 1D and the axis is set to 0.

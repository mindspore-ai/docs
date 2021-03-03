# Function Differences of AvgPool2d

PyTorch: Performs average pooling for H and W dimensions of the input data. You only need to specify the desired shape of the H and W dimensions of data after pooling. It is unnecessary to manually calculate and specify the `kernel_size`, `stride`, etc.

MindSpore：The user needs to manually calculate and specify the `kernel_size`, `stride`, etc.

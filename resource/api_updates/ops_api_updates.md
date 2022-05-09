# API Updates

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.TensorShape](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.TensorShape.html)|New|Returns the shape of the input tensor.|r1.7: Ascend/GPU/CPU|Array Operation
|[mindspore.ops.unique](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.unique.html)|New|Returns the unique elements of input tensor and also return a tensor containing the index of each value of input tensor corresponding to the output unique tensor.|r1.7: Ascend/GPU/CPU|Array Operation
|[mindspore.ops.ms_hybrid](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.ms_hybrid.html)|New|The decorator of the Hybrid DSL function for the Custom Op.|r1.7: Ascend/GPU/CPU|Decorators
|[mindspore.ops.Addcdiv](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Addcdiv.html)|New|Performs the element-wise division of tensor x1 by tensor x2, multiply the result by the scalar value and add it to input_data.|r1.7: Ascend|Element-by-Element Operation
|[mindspore.ops.Addcmul](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Addcmul.html)|New|Performs the element-wise product of tensor x1 and tensor x2, multiply the result by the scalar value and add it to input_data.|r1.7: Ascend|Element-by-Element Operation
|[mindspore.ops.Einsum](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Einsum.html)|New|This operator uses equation to represent a tuple of tensors operations, you can use this operator to perform diagonal/reducesum/transpose/matmul/mul/inner product operations, etc.|r1.7: GPU|Element-by-Element Operation
|[mindspore.ops.absolute](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.absolute.html)|New|Refer to mindspore.ops.Abs.|r1.7: Ascend/GPU/CPU|Element-by-Element Operations
|[mindspore.ops.arange](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.arange.html)|New|Returns evenly spaced values within a given interval.|r1.7: Ascend/GPU/CPU|operations--Other Operators
|[mindspore.ops.DynamicShape](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.DynamicShape.html)|Changed|Same as operator TensorShape.|r1.6: Ascend/GPU/CPU => r1.7: Deprecated|Array Operation
|[mindspore.ops.Conj](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Conj.html)|Changed|Returns a tensor of complex numbers that are the complex conjugate of each element in input.|r1.6: GPU => r1.7: CPU/GPU|Element-by-Element Operation
|[mindspore.ops.Imag](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Imag.html)|Changed|Returns a new tensor containing imaginary value of the input.|r1.6: GPU => r1.7: CPU/GPU|Element-by-Element Operation
|[mindspore.ops.MulNoNan](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.MulNoNan.html)|Changed|Computes x * y element-wise.|r1.6: Ascend => r1.7: Ascend/CPU|Element-by-Element Operation
|[mindspore.ops.Real](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Real.html)|Changed|Returns a Tensor that is the real part of the input.|r1.6: GPU => r1.7: CPU/GPU|Element-by-Element Operation
|[mindspore.ops.Rsqrt](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Rsqrt.html)|Changed|Computes reciprocal of square root of input tensor element-wise.|r1.6: Ascend/GPU => r1.7: Ascend/GPU/CPU|Element-by-Element Operation
|[mindspore.ops.Softplus](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Softplus.html)|Changed|Softplus activation function.|r1.6:   /GPU/CPU => r1.7: Ascend/GPU/CPU|Activation Function
|[mindspore.ops.Tanh](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Tanh.html)|Changed|Tanh activation function.|r1.6: GPU/  /CPU => r1.7: Ascend/GPU/CPU|Activation Function
|[mindspore.ops.NLLLoss](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.NLLLoss.html)|Changed|Gets the negative log likelihood loss between logits and labels.|r1.6: Ascend/GPU => r1.7: Ascend/GPU/CPU|Loss Function
|[mindspore.ops.AvgPool3D](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.AvgPool3D.html)|Changed|3D Average pooling operation.|r1.6: Ascend => r1.7: Ascend/CPU|Neural Network
|[mindspore.ops.MaxPool3D](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.MaxPool3D.html)|Changed|3D max pooling operation.|r1.6: Ascend/GPU => r1.7: Ascend/GPU/CPU|Neural Network
|[mindspore.ops.ApplyGradientDescent](https://www.mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.ApplyGradientDescent.html)|Changed|Updates var by subtracting alpha * delta from it.|r1.6:   /GPU => r1.7: Ascend/GPU|Optimizer

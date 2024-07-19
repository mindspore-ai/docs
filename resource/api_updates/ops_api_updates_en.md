# mindspore.ops.primitive API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, is shown in the following table.

## Differences between 2.3.0 and 2.3.0rc2

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.Barrier](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.Barrier.html#mindspore.ops.Barrier)|New|Synchronizes all processes in the specified group.|r2.3.0: Ascend|Communication Operator
|[mindspore.ops.CollectiveGather](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.CollectiveGather.html#mindspore.ops.CollectiveGather)|New|Gathers tensors from the specified communication group.|r2.3.0: Ascend|Communication Operator
|[mindspore.ops.CollectiveScatter](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.CollectiveScatter.html#mindspore.ops.CollectiveScatter)|New|Scatter tensor evently across the processes in the specified communication group.|r2.3.0: Ascend|Communication Operator
|[mindspore.ops.Receive](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.Receive.html#mindspore.ops.Receive)|New|Receive tensors from src_rank.|r2.3.0: Ascend/GPU|Communication Operator
|[mindspore.ops.Reduce](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.Reduce.html#mindspore.ops.Reduce)|New|Reduces tensors across the processes in the specified communication group, sends the result to the target dest_rank(local rank), and returns the tensor which is sent to the target process.|r2.3.0: Ascend|Communication Operator
|[mindspore.ops.Send](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.Send.html#mindspore.ops.Send)|New|Send tensors to the specified dest_rank.|r2.3.0: Ascend/GPU|Communication Operator
|[mindspore.ops.Custom](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.Custom.html#mindspore.ops.Custom)|Changed|Custom primitive is used for user defined operators and is to enhance the expressive ability of built-in primitives.|r2.3.0rc2: GPU/CPU => r2.3.0: GPU/CPU/ASCEND|Customizing Operator

## Differences between 2.3.0rc2 and 2.3.0rc1

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.Custom](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.Custom.html#mindspore.ops.Custom)|Changed|Custom primitive is used for user defined operators and is to enhance the expressive ability of built-in primitives.|r2.3.0rc1: Ascend/GPU/CPU => r2.3.0rc2: GPU/CPU|Customizing Operator
|[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Deleted|Please use mindspore.ops.MaxPoolWithArgmaxV2 instead.|r2.3.0rc1: |Neural Network
|[mindspore.ops.MaxPoolWithArgmaxV2](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmaxV2.html#mindspore.ops.MaxPoolWithArgmaxV2)|Deleted|Performs max pooling on the input Tensor and returns both max values and indices.|r2.3.0rc1: Ascend/GPU/CPU|Neural Network
|[mindspore.ops.Zeros](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.Zeros.html#mindspore.ops.Zeros)|Changed|Zeros will be deprecated in the future.|r2.3.0rc1:  => r2.3.0rc2: Ascend/GPU/CPU|Tensor Construction

## Differences between 2.3.0rc1 and 2.2.14

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.silent_check.ASDBase](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.silent_check.ASDBase.html#mindspore.ops.silent_check.ASDBase)|Changed|r2.2: ASD Base Class. => r2.3.0rc1: ASDBase is the base class of operator with accuracy-sensitive detection feature in python.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: Ascend|Accuracy-Sensitive Detection
|[mindspore.ops.BatchToSpaceND](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.BatchToSpaceND.html#mindspore.ops.BatchToSpaceND)|Changed|Please use batch_to_space_nd instead.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Array Operation
|[mindspore.ops.Identity](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.Identity.html#mindspore.ops.Identity)|Changed|r2.2: The mindspore.ops.Identity interface is deprecated, please use the mindspore.ops.deepcopy() instead. => r2.3.0rc1: Refer to mindspore.ops.deepcopy() for more details.|r2.2:  => r2.3.0rc1: Ascend/GPU/CPU|Array Operation
|[mindspore.ops.InplaceUpdate](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.InplaceUpdate.html#mindspore.ops.InplaceUpdate)|Changed|r2.2: Please use InplaceUpdateV2 instead. => r2.3.0rc1: Please use mindspore.ops.InplaceUpdateV2 instead.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Array Operation
|[mindspore.ops.TensorDump](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.TensorDump.html#mindspore.ops.TensorDump)|New|Save the Tensor as an npy file in numpy format.|r2.2: r2.3.0rc1: Ascend|Debugging Operator
|[mindspore.ops.ExtractVolumePatches](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.ExtractVolumePatches.html#mindspore.ops.ExtractVolumePatches)|Changed|r2.2: Extract patches from input and put them in the "depth" output dimension. => r2.3.0rc1: ops.ExtractVolumePatches is deprecated from version 2.3 and will be removed in a future version.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Image Processing
|[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Changed|r2.2: Please use MaxPoolWithArgmaxV2 instead. => r2.3.0rc1: Please use mindspore.ops.MaxPoolWithArgmaxV2 instead.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Neural Network
|[mindspore.ops.ResizeBilinear](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.ResizeBilinear.html#mindspore.ops.ResizeBilinear)|Deleted|This API is deprecated, please use the mindspore.ops.ResizeBilinearV2 instead.||Neural Network
|[mindspore.ops.ScatterNonAliasingAdd](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html#mindspore.ops.ScatterNonAliasingAdd)|Changed|Please use TensorScatterAdd instead.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Parameter Operation Operator
|[mindspore.ops.RandpermV2](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.RandpermV2.html#mindspore.ops.RandpermV2)|Changed|r2.2: Generates random permutation of integers from 0 to n-1 without repeating. => r2.3.0rc1: Refer to mindspore.ops.randperm() for more details.|r2.2: Ascend/CPU => r2.3.0rc1: CPU|Random Generation Operator
|[mindspore.ops.ScalarCast](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.ScalarCast.html#mindspore.ops.ScalarCast)|Changed|r2.2: Casts the input scalar to another type. => r2.3.0rc1: 'ops.ScalarCast' is deprecated from version 2.3 and will be removed in a future version, please use int(x) or float(x) instead.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Type Conversion

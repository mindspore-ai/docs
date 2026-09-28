# MindSpore Lite Supported Hardware Backend List

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/docs/source_en/reference/operator_list_lite.md)

## Abs

### Function

Computes the absolute value element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## AbsGrad

### Function

Computes the gradient of the absolute value function.

### Supported Hardware Backends

- CPU: FP32

---

## Activation

### Function

Activation function.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## ActivationGrad

### Function

Computes the gradient of a specific activation function.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Adam

### Function

Performs a single parameter update step of the Adam optimizer.

### Supported Hardware Backends

- CPU: FP32

---

## AddFusion

### Function

Computes addition element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int8
- Ascend: FP16

---

## AdderFusion

### Function

Addition-based convolution operation.

### Supported Hardware Backends

- CPU: FP32

---

## AddGrad

### Function

Computes the gradient of the addition operation.

### Supported Hardware Backends

- CPU: FP32

---

## AddN

### Function

Performs element-wise addition of N input tensors with the same shape and data type.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Affine

### Function

Applies an affine transformation to the input tensor.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## All

### Function

Determines whether all elements in the tensor are True (non-zero) along the specified dimension.

### Supported Hardware Backends

- CPU: FP32

---

## AllGather

### Function

Distributed collective communication operation.

### Supported Hardware Backends

- CPU: FP32

---

## ApplyMomentum

### Function

Performs a single parameter update step of stochastic gradient descent with momentum.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## ArgmaxFusion

### Function

Finds the maximum value along a certain dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## ArgminFusion

### Function

Finds the minimum value along a certain dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Assert

### Function

Assertion.

### Supported Hardware Backends

- CPU: FP16, FP32, Bool

---

## Assign

### Function

Assigns a value to a variable.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## AvgPoolFusion

### Function

Average pooling.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## AvgPoolGrad

### Function

Computes the gradient of the average pooling layer.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## BatchNorm

### Function

Batch normalization.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## BatchNormGrad

### Function

Computes the gradient of the batch normalization layer.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## BatchToSpace

### Function

The inverse operation of space-to-batch transformation.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

---

## BatchToSpaceND

### Function

The ND general version of BatchToSpace.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

---

## BiasAdd

### Function

Adds a bias vector to the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## BiasAddGrad

### Function

Computes the gradient of the BiasAdd operation.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## BinaryCrossEntropy

### Function

Computes the binary cross-entropy loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## BinaryCrossEntropyGrad

### Function

Computes the gradient of the binary cross-entropy loss function.

### Supported Hardware Backends

- CPU: FP32

---

## BroadcastTo

### Function

Expands/broadcasts a tensor along specified dimensions to a larger shape according to NumPy broadcasting rules.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

---

## Call

### Function

Calls a subgraph or function.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Ascend: FP16

---

## Cast

### Function

Data type conversion.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Ceil

### Function

Rounds up to the nearest integer.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Clip

### Function

Clips the range of elements.

### Supported Hardware Backends

- CPU: FP32, Int32
- Ascend: FP16

---

## Concat

### Function

Concatenates tensors.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

---

## ConstantOfShape

### Function

Generates a tensor with the same shape as the input, filled with a specified constant.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## Conv2DFusion

### Function

2D convolution.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Conv2DBackpropFilterFusion

### Function

Computes the gradient of a normal convolution operation with respect to the convolution kernel.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Conv2DBackpropInputFusion

### Function

Computes the gradient of a normal convolution operation with respect to the input data.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Conv2dTransposeFusion

### Function

Performs transposed convolution.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Cos

### Function

Computes the cosine element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Crop

### Function

Crops a specified region from the input image or feature map.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8

---

## CropAndResize

### Function

Crops regions from the input image according to a set of bounding boxes, then resizes each region to a uniform size.

### Supported Hardware Backends

- CPU: FP32
- Kirin NPU: FP16

---

## CumSum

### Function

Cumulative sum of elements.

### Supported Hardware Backends

- CPU: FP32, Int32
- Ascend: FP16

---

## CustomExtractFeatures

### Function

Custom feature extraction operator.

### Supported Hardware Backends

- CPU: FP32

---

## CustomNormalize

### Function

Custom normalization operator.

### Supported Hardware Backends

- CPU: FP32

---

## CustomPredict

### Function

Custom prediction operator.

### Supported Hardware Backends

- CPU: FP32, Int32

---

## DEConv2DGradFilter

### Function

Computes the gradient of transposed convolution with respect to the convolution kernel.

### Supported Hardware Backends

- CPU: FP32

---

## DepthToSpace

### Function

Rearranges depth data into spatial dimensions.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

---

## DetectionPostProcess

### Function

Object detection post-processing.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8

---

## DivFusion

### Function

Element-wise division.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## DivGrad

### Function

Computes the gradient of the division operation.

### Supported Hardware Backends

- CPU: FP32

---

## Dropout

### Function

Randomly sets some elements of the input tensor to 0.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

---

## DropoutGrad

### Function

Computes the gradient of the Dropout operation.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## DynamicQuant

### Function

Dynamically quantizes a floating-point tensor to uint8 type.

### Supported Hardware Backends

- CPU: FP32

---

## Eltwise

### Function

Element-wise operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Elu

### Function

Activation function that applies exponential correction to negative inputs.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

---

## Equal

### Function

Determines whether the inputs are equal.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## EmbeddingLookupFusion

### Function

Optimized word embedding lookup that maps integer indices to dense vectors.

### Supported Hardware Backends

- CPU: FP32

---

## Erf

### Function

Error function.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

---

## ExpFusion

### Function

Computes the exponential element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## ExpandDims

### Function

Inserts a dimension of length 1 at the specified position.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

---

## Fill

### Function

Generates a tensor filled with a specified constant.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Flatten

### Function

Flattens data along dimensions.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

---

## FlattenGrad

### Function

Computes the gradient of the Flatten operation.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Floor

### Function

Rounds down to the nearest integer.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## FloorDiv

### Function

Element-wise floor division.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## FloorMod

### Function

Element-wise modulo operation, where the sign of the result matches the divisor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## FullConnection

### Function

Fully connected layer.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## FusedBatchNorm

### Function

Normalizes the input.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- Ascend: FP16

---

## Gather

### Function

Collects elements at specified index positions along a single dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

---

## GatherD

### Function

Collects elements from the input tensor according to an index tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Ascend: FP16

---

## GatherNd

### Function

Collects elements at specified positions from the input tensor according to an index tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## GLU

### Function

Gated linear unit activation function that splits the input into two parts and multiplies them element-wise.

### Supported Hardware Backends

- CPU: FP32

---

## Greater

### Function

Compares two tensors element-wise and returns the logical result of A > B (True/False).

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## GreaterEqual

### Function

Compares two tensors element-wise and returns the logical result of A ≥ B (True/False).

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## GroupNormFusion

### Function

Fused optimized group normalization.

### Supported Hardware Backends

- CPU: FP32

---

## GRU

### Function

Gated recurrent unit, a simplified version of LSTM.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## HashtableLookup

### Function

Hash table lookup.

### Supported Hardware Backends

- CPU: FP32, Int32

---

## InstanceNorm

### Function

Instance normalization.

### Supported Hardware Backends

- CPU: FP16, FP32
- Kirin NPU: FP16
- Ascend: FP16

---

## InvertPermutation

### Function

Inverts permutation indices.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## IsFinite

### Function

Checks whether each element in the tensor is a finite value (not inf/NaN).

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## L2NormalizeFusion

### Function

Fused optimized L2 normalization.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8

---

## LayerNormFusion

### Function

Fused optimized layer normalization.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## LayerNormGrad

### Function

Computes the gradient of layer normalization.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## LeakyReLU

### Function

Leaky ReLU activation function that gives a small slope to negative inputs.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Less

### Function

Compares two tensors element-wise and returns the logical result of A < B.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## LessEqual

### Function

Compares A ≤ B element-wise and returns a boolean tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## LRN

### Function

Local response normalization.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## Log

### Function

Computes the logarithm element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Log1p

### Function

Computes log(1+X).

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## LogGrad

### Function

Computes the gradient of the logarithmic function.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## LogicalAnd

### Function

Element-wise logical AND operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## LogicalNot

### Function

Element-wise logical NOT operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## LogicalOr

### Function

Element-wise logical OR operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## LogSoftmax

### Function

Applies softmax to the input vector, then takes the logarithm of the softmax result.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

---

## LshProjection

### Function

Locality-sensitive hashing projection.

### Supported Hardware Backends

- CPU: FP32

---

## LSTM

### Function

Long short-term memory network unit.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## LSTMGrad

### Function

Computes the backpropagation gradient of LSTM with respect to the hidden state.

### Supported Hardware Backends

- CPU: FP32

---

## LSTMGradData

### Function

Computes the backpropagation gradient of LSTM with respect to the input data.

### Supported Hardware Backends

- CPU: FP32

---

## LSTMGradWeight

### Function

Computes the backpropagation gradient of LSTM with respect to the weights.

### Supported Hardware Backends

- CPU: FP32

---

## MatMulFusion

### Function

Performs matrix multiplication on 2 inputs.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Maximum

### Function

Takes the element-wise maximum value.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## MaximumGrad

### Function

Computes the gradient of the maximum function.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## MaxPoolFusion

### Function

Max pooling.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## MaxPoolGrad

### Function

Computes the gradient of the max pooling layer.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Merge

### Function

Obtains the first available tensor from a set of input tensors and outputs it.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Minimum

### Function

Takes the element-wise minimum value.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## MinimumGrad

### Function

Computes the gradient of the minimum function.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Mod

### Function

Returns the remainder of division elements.

### Supported Hardware Backends

- CPU: FP32, Int32
- Ascend: FP16

---

## MulFusion

### Function

Element-wise multiplication.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## MulGrad

### Function

Computes the gradient of the multiplication operation.

### Supported Hardware Backends

- CPU: FP32

---

## Neg

### Function

Computes the negative value element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## NegGrad

### Function

Computes the gradient of the negation operation.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## NLLLoss

### Function

Computes the negative log-likelihood loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## NLLLossGrad

### Function

Computes the gradient of NLLLoss.

### Supported Hardware Backends

- CPU: FP32

---

## NotEqual

### Function

Compares two tensors element-wise and returns the logical result of A != B.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## NonMaxSuppression

### Function

Non-maximum suppression.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## NonZero

### Function

Returns the indices of all non-zero elements in the input tensor.

### Supported Hardware Backends

- CPU: Bool
- Ascend: FP16

---

## OneHot

### Function

Converts an integer index tensor into a one-hot encoding representation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- GPU (Mali/Adreno): FP16, FP32, Int32

---

## OnesLike

### Function

Creates a new tensor with exactly the same shape as the input tensor X but with all elements set to 1.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

---

## PadFusion

### Function

Pads the input tensor with the specified padding to reach the specified size.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## PartialFusion

### Function

Partial fusion.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

---

## PowFusion

### Function

Computes the power element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## PowerGrad

### Function

Computes the gradient of the power operation.

### Supported Hardware Backends

- CPU: FP32

---

## PriorBox

### Function

Generates prior boxes.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8
- Ascend: FP16

---

## PReLUFusion

### Function

PReLU activation function.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## QuantDTypeCast

### Function

Performs quantized data type conversion.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8

---

## RaggedRange

### Function

Generates a sequence with non-uniform intervals.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## RandomNormal

### Function

Generates a tensor whose values are randomly sampled from a normal distribution.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## RandomStandardNormal

### Function

Generates a random number tensor that follows the standard normal distribution.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Range

### Function

Generates elements within a certain interval.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

---

## Rank

### Function

Returns the number of dimensions of the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## RealDiv

### Function

Element-wise division.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

---

## Reciprocal

### Function

Returns the reciprocal.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8
- Kirin NPU: FP16
- Ascend: FP16

---

## ReduceFusion

### Function

Reduction operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## ReduceScatter

### Function

Distributed operation that segments the input tensor and distributes it to each device, with each device retaining only one segment of the result.

### Supported Hardware Backends

- CPU: FP32

---

## Reshape

### Function

Changes the tensor shape while keeping the total number of elements unchanged.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

---

## Resize

### Function

Upsamples or resizes the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## ResizeGrad

### Function

Computes the gradient of Resize.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## ReverseV2

### Function

Reverses the tensor along the specified axis.

### Supported Hardware Backends

- CPU: FP32, Int32

---

## ReverseSequence

### Function

Partially reverses variable-length sequences of the input tensor.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## ROIPooling

### Function

Region of interest pooling.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## Round

### Function

Rounds to the nearest integer value.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Rsqrt

### Function

Computes the reciprocal of the square root element-wise, used for normalization.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

---

## RsqrtGrad

### Function

Computes the gradient of the reciprocal square root.

### Supported Hardware Backends

- CPU: FP32

---

## Select

### Function

Selects elements from two tensors based on a condition.

### Supported Hardware Backends

- CPU: FP32, Bool

---

## Selu

### Function

Self-normalizing exponential linear unit activation function.

### Supported Hardware Backends

None

---

## ScaleFusion

### Function

Fuses the scaling operation with adjacent operators.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## ScatterNd

### Function

Scatters values from the update tensor to specified positions in the output tensor according to indices.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

---

## ScatterNdUpdate

### Function

Updates the values of the input data using the given values and input indices.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## SGD

### Function

Stochastic gradient descent optimizer.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## Shape

### Function

Obtains the tensor shape.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SigmoidCrossEntropyWithLogits

### Function

Combines Sigmoid activation and cross-entropy loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## SigmoidCrossEntropyWithLogitsGrad

### Function

Computes the gradient of the cross-entropy loss with Sigmoid.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## Sin

### Function

Computes the sine element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## Size

### Function

Obtains the tensor dimension size.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

---

## SliceFusion

### Function

Tensor slicing operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SkipGram

### Function

The core operation of the Skip-gram model, used for word vector training.

### Supported Hardware Backends

- CPU: FP32

---

## SmoothL1Loss

### Function

Smooth L1 loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## SmoothL1LossGrad

### Function

Computes the gradient of the smooth L1 loss.

### Supported Hardware Backends

- CPU: FP32

---

## Softmax

### Function

Normalization operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SoftmaxGrad

### Function

Computes the gradient of Softmax.

### Supported Hardware Backends

- CPU: FP32

---

## Softplus

### Function

Smooth variant of ReLU.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

---

## SpaceToBatch

### Function

Moves values from height and width dimensions to the depth dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SpaceToBatchND

### Function

Splits blocks of data from spatial dimensions into the batch dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

---

## SpaceToDepth

### Function

Reorganizes spatial data into depth channels.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32

---

## SparseToDense

### Function

Converts a sparse representation to a dense tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- GPU (Mali/Adreno): FP16, FP32, Int32

---

## SparseSoftmaxCrossEntropyWithLogits

### Function

Softmax cross-entropy with sparse labels.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

---

## Splice

### Function

Concatenates multiple slices or ranges of the input tensor along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Split

### Function

Splits the input tensor into multiple smaller output tensors along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SplitWithOverlap

### Function

Splits the tensor with overlap.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Sqrt

### Function

Computes the square root element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SqrtGrad

### Function

Computes the gradient of the square root.

### Supported Hardware Backends

- CPU: FP32

---

## Square

### Function

Computes the square element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SquaredDifference

### Function

Computes (A-B)² element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32

---

## Squeeze

### Function

Removes dimensions of size 1.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- GPU (Mali/Adreno): FP16, FP32, Int32

---

## StridedSlice

### Function

Tensor slicing.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## StridedSliceGrad

### Function

Computes the gradient of the slicing operation.

### Supported Hardware Backends

- CPU: FP16, FP32

---

## Stack

### Function

Stacks multiple tensors along a new axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SubFusion

### Function

Element-wise subtraction.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## SubGrad

### Function

Computes the gradient of subtraction.

### Supported Hardware Backends

- CPU: FP32

---

## Switch

### Function

Selects an output branch based on a boolean condition.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

---

## SwitchLayer

### Function

Selects different sub-network branches for execution in the model.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

---

## TensorListFromTensor

### Function

Converts an ordinary tensor into a tensor list, splitting along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## TensorListGetItem

### Function

Retrieves the tensor at the specified index position from a tensor list.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## TensorListReserve

### Function

Pre-allocates an empty tensor list with the specified element data type and initial capacity.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## TensorListSetItem

### Function

Inserts a tensor at the specified position in a tensor list.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## TensorListStack

### Function

Stacks a tensor list into an ordinary tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## TensorScatterAdd

### Function

Scatter-adds the values of the update tensor to specified positions in the target tensor according to indices.

### Supported Hardware Backends

- CPU: FP32, Int32

---

## TileFusion

### Function

Tiles a given matrix.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Kirin NPU: FP16
- Ascend: FP16

---

## TopKFusion

### Function

Returns the top K elements from the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Ascend: FP16

---

## Transpose

### Function

Tensor transposition.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

---

## UniformReal

### Function

Generates a random number tensor that follows a uniform distribution.

### Supported Hardware Backends

- CPU: FP32, Int32

---

## Unique

### Function

Returns the unique values in the input tensor, and optionally returns the indices and counts of those values.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## UnsortedSegmentSum

### Function

Performs segmented summation on a tensor without requiring the segment indices to be ordered.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## Unsqueeze

### Function

Adds a new dimension to the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32

---

## Unstack

### Function

Splits the tensor into multiple sub-tensors along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

---

## Where

### Function

Element selection.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

---

## ZerosLike

### Function

Generates a new tensor with the same shape as the input tensor but filled with zeros.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

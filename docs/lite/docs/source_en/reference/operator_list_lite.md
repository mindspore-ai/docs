# MindSpore Lite Supported Hardware Backend List

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.11.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.11.0/docs/lite/docs/source_en/reference/operator_list_lite.md)

## Abs

### Function

Computes the absolute value element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## AbsGrad

### Function

Computes the gradient of the absolute value function.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Activation

### Function

Activation function.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## ActivationGrad

### Function

Computes the gradient of a specific activation function.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Adam

### Function

Performs a single parameter update step of the Adam optimizer.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## AddFusion

### Function

Computes addition element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int8
- Ascend: FP16

</br></br>

---

## AdderFusion

### Function

Addition-based convolution operation.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## AddGrad

### Function

Computes the gradient of the addition operation.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## AddN

### Function

Performs element-wise addition of N input tensors with the same shape and data type.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Affine

### Function

Applies an affine transformation to the input tensor.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## All

### Function

Determines whether all elements in the tensor are True (non-zero) along the specified dimension.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## AllGather

### Function

Distributed collective communication operation.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## ApplyMomentum

### Function

Performs a single parameter update step of stochastic gradient descent with momentum.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## ArgmaxFusion

### Function

Finds the maximum value along a certain dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## ArgminFusion

### Function

Finds the minimum value along a certain dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Assert

### Function

Assertion.

### Supported Hardware Backends

- CPU: FP16, FP32, Bool

</br></br>

---

## Assign

### Function

Assigns a value to a variable.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## AvgPoolFusion

### Function

Average pooling.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## AvgPoolGrad

### Function

Computes the gradient of the average pooling layer.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## BatchNorm

### Function

Batch normalization.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## BatchNormGrad

### Function

Computes the gradient of the batch normalization layer.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## BatchToSpace

### Function

The inverse operation of space-to-batch transformation.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## BatchToSpaceND

### Function

The ND general version of BatchToSpace.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## BiasAdd

### Function

Adds a bias vector to the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## BiasAddGrad

### Function

Computes the gradient of the BiasAdd operation.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## BinaryCrossEntropy

### Function

Computes the binary cross-entropy loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## BinaryCrossEntropyGrad

### Function

Computes the gradient of the binary cross-entropy loss function.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## BroadcastTo

### Function

Expands/broadcasts a tensor along specified dimensions to a larger shape according to NumPy broadcasting rules.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

</br></br>

---

## Call

### Function

Calls a subgraph or function.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Ascend: FP16

</br></br>

---

## Cast

### Function

Data type conversion.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Ceil

### Function

Rounds up to the nearest integer.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Clip

### Function

Clips the range of elements.

### Supported Hardware Backends

- CPU: FP32, Int32
- Ascend: FP16

</br></br>

---

## Concat

### Function

Concatenates tensors.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## ConstantOfShape

### Function

Generates a tensor with the same shape as the input, filled with a specified constant.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## Conv2DFusion

### Function

2D convolution.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Conv2DBackpropFilterFusion

### Function

Computes the gradient of a normal convolution operation with respect to the convolution kernel.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Conv2DBackpropInputFusion

### Function

Computes the gradient of a normal convolution operation with respect to the input data.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Conv2dTransposeFusion

### Function

Performs transposed convolution.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Cos

### Function

Computes the cosine element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Crop

### Function

Crops a specified region from the input image or feature map.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8

</br></br>

---

## CropAndResize

### Function

Crops regions from the input image according to a set of bounding boxes, then resizes each region to a uniform size.

### Supported Hardware Backends

- CPU: FP32
- Kirin NPU: FP16

</br></br>

---

## CumSum

### Function

Cumulative sum of elements.

### Supported Hardware Backends

- CPU: FP32, Int32
- Ascend: FP16

</br></br>

---

## CustomExtractFeatures

### Function

Custom feature extraction operator.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## CustomNormalize

### Function

Custom normalization operator.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## CustomPredict

### Function

Custom prediction operator.

### Supported Hardware Backends

- CPU: FP32, Int32

</br></br>

---

## DEConv2DGradFilter

### Function

Computes the gradient of transposed convolution with respect to the convolution kernel.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## DepthToSpace

### Function

Rearranges depth data into spatial dimensions.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## DetectionPostProcess

### Function

Object detection post-processing.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8

</br></br>

---

## DivFusion

### Function

Element-wise division.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## DivGrad

### Function

Computes the gradient of the division operation.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Dropout

### Function

Randomly sets some elements of the input tensor to 0.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

</br></br>

---

## DropoutGrad

### Function

Computes the gradient of the Dropout operation.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## DynamicQuant

### Function

Dynamically quantizes a floating-point tensor to uint8 type.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Eltwise

### Function

Element-wise operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Elu

### Function

Activation function that applies exponential correction to negative inputs.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

</br></br>

---

## Equal

### Function

Determines whether the inputs are equal.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## EmbeddingLookupFusion

### Function

Optimized word embedding lookup that maps integer indices to dense vectors.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Erf

### Function

Error function.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

</br></br>

---

## ExpFusion

### Function

Computes the exponential element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## ExpandDims

### Function

Inserts a dimension of length 1 at the specified position.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## Fill

### Function

Generates a tensor filled with a specified constant.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Flatten

### Function

Flattens data along dimensions.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## FlattenGrad

### Function

Computes the gradient of the Flatten operation.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Floor

### Function

Rounds down to the nearest integer.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## FloorDiv

### Function

Element-wise floor division.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## FloorMod

### Function

Element-wise modulo operation, where the sign of the result matches the divisor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## FullConnection

### Function

Fully connected layer.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## FusedBatchNorm

### Function

Normalizes the input.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- Ascend: FP16

</br></br>

---

## Gather

### Function

Collects elements at specified index positions along a single dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## GatherD

### Function

Collects elements from the input tensor according to an index tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Ascend: FP16

</br></br>

---

## GatherNd

### Function

Collects elements at specified positions from the input tensor according to an index tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## GLU

### Function

Gated linear unit activation function that splits the input into two parts and multiplies them element-wise.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Greater

### Function

Compares two tensors element-wise and returns the logical result of A > B (True/False).

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## GreaterEqual

### Function

Compares two tensors element-wise and returns the logical result of A ≥ B (True/False).

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## GroupNormFusion

### Function

Fused optimized group normalization.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## GRU

### Function

Gated recurrent unit, a simplified version of LSTM.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## HashtableLookup

### Function

Hash table lookup.

### Supported Hardware Backends

- CPU: FP32, Int32

</br></br>

---

## InstanceNorm

### Function

Instance normalization.

### Supported Hardware Backends

- CPU: FP16, FP32
- Kirin NPU: FP16
- Ascend: FP16

</br></br>

---

## InvertPermutation

### Function

Inverts permutation indices.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## IsFinite

### Function

Checks whether each element in the tensor is a finite value (not inf/NaN).

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## L2NormalizeFusion

### Function

Fused optimized L2 normalization.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8

</br></br>

---

## LayerNormFusion

### Function

Fused optimized layer normalization.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## LayerNormGrad

### Function

Computes the gradient of layer normalization.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## LeakyReLU

### Function

Leaky ReLU activation function that gives a small slope to negative inputs.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Less

### Function

Compares two tensors element-wise and returns the logical result of A < B.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## LessEqual

### Function

Compares A ≤ B element-wise and returns a boolean tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## LRN

### Function

Local response normalization.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## Log

### Function

Computes the logarithm element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Log1p

### Function

Computes log(1+X).

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## LogGrad

### Function

Computes the gradient of the logarithmic function.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## LogicalAnd

### Function

Element-wise logical AND operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## LogicalNot

### Function

Element-wise logical NOT operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## LogicalOr

### Function

Element-wise logical OR operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## LogSoftmax

### Function

Applies softmax to the input vector, then takes the logarithm of the softmax result.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

</br></br>

---

## LshProjection

### Function

Locality-sensitive hashing projection.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## LSTM

### Function

Long short-term memory network unit.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## LSTMGrad

### Function

Computes the backpropagation gradient of LSTM with respect to the hidden state.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## LSTMGradData

### Function

Computes the backpropagation gradient of LSTM with respect to the input data.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## LSTMGradWeight

### Function

Computes the backpropagation gradient of LSTM with respect to the weights.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## MatMulFusion

### Function

Performs matrix multiplication on 2 inputs.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Maximum

### Function

Takes the element-wise maximum value.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## MaximumGrad

### Function

Computes the gradient of the maximum function.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## MaxPoolFusion

### Function

Max pooling.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## MaxPoolGrad

### Function

Computes the gradient of the max pooling layer.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Merge

### Function

Obtains the first available tensor from a set of input tensors and outputs it.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Minimum

### Function

Takes the element-wise minimum value.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## MinimumGrad

### Function

Computes the gradient of the minimum function.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Mod

### Function

Returns the remainder of division elements.

### Supported Hardware Backends

- CPU: FP32, Int32
- Ascend: FP16

</br></br>

---

## MulFusion

### Function

Element-wise multiplication.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## MulGrad

### Function

Computes the gradient of the multiplication operation.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Neg

### Function

Computes the negative value element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## NegGrad

### Function

Computes the gradient of the negation operation.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## NLLLoss

### Function

Computes the negative log-likelihood loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## NLLLossGrad

### Function

Computes the gradient of NLLLoss.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## NotEqual

### Function

Compares two tensors element-wise and returns the logical result of A != B.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## NonMaxSuppression

### Function

Non-maximum suppression.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## NonZero

### Function

Returns the indices of all non-zero elements in the input tensor.

### Supported Hardware Backends

- CPU: Bool
- Ascend: FP16

</br></br>

---

## OneHot

### Function

Converts an integer index tensor into a one-hot encoding representation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- GPU (Mali/Adreno): FP16, FP32, Int32

</br></br>

---

## OnesLike

### Function

Creates a new tensor with exactly the same shape as the input tensor X but with all elements set to 1.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## PadFusion

### Function

Pads the input tensor with the specified padding to reach the specified size.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## PartialFusion

### Function

Partial fusion.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

</br></br>

---

## PowFusion

### Function

Computes the power element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## PowerGrad

### Function

Computes the gradient of the power operation.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## PriorBox

### Function

Generates prior boxes.

### Supported Hardware Backends

- CPU: FP32, Int8, UInt8
- Ascend: FP16

</br></br>

---

## PReLUFusion

### Function

PReLU activation function.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## QuantDTypeCast

### Function

Performs quantized data type conversion.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8

</br></br>

---

## RaggedRange

### Function

Generates a sequence with non-uniform intervals.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## RandomNormal

### Function

Generates a tensor whose values are randomly sampled from a normal distribution.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## RandomStandardNormal

### Function

Generates a random number tensor that follows the standard normal distribution.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Range

### Function

Generates elements within a certain interval.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## Rank

### Function

Returns the number of dimensions of the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## RealDiv

### Function

Element-wise division.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

</br></br>

---

## Reciprocal

### Function

Returns the reciprocal.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8
- Kirin NPU: FP16
- Ascend: FP16

</br></br>

---

## ReduceFusion

### Function

Reduction operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## ReduceScatter

### Function

Distributed operation that segments the input tensor and distributes it to each device, with each device retaining only one segment of the result.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Reshape

### Function

Changes the tensor shape while keeping the total number of elements unchanged.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## Resize

### Function

Upsamples or resizes the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## ResizeGrad

### Function

Computes the gradient of Resize.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## ReverseV2

### Function

Reverses the tensor along the specified axis.

### Supported Hardware Backends

- CPU: FP32, Int32

</br></br>

---

## ReverseSequence

### Function

Partially reverses variable-length sequences of the input tensor.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## ROIPooling

### Function

Region of interest pooling.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## Round

### Function

Rounds to the nearest integer value.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Rsqrt

### Function

Computes the reciprocal of the square root element-wise, used for normalization.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## RsqrtGrad

### Function

Computes the gradient of the reciprocal square root.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Select

### Function

Selects elements from two tensors based on a condition.

### Supported Hardware Backends

- CPU: FP32, Bool

</br></br>

---

## Selu

### Function

Self-normalizing exponential linear unit activation function.

### Supported Hardware Backends

None

</br></br>

---

## ScaleFusion

### Function

Fuses the scaling operation with adjacent operators.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## ScatterNd

### Function

Scatters values from the update tensor to specified positions in the output tensor according to indices.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## ScatterNdUpdate

### Function

Updates the values of the input data using the given values and input indices.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## SGD

### Function

Stochastic gradient descent optimizer.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## Shape

### Function

Obtains the tensor shape.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SigmoidCrossEntropyWithLogits

### Function

Combines Sigmoid activation and cross-entropy loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## SigmoidCrossEntropyWithLogitsGrad

### Function

Computes the gradient of the cross-entropy loss with Sigmoid.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## Sin

### Function

Computes the sine element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## Size

### Function

Obtains the tensor dimension size.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- Ascend: FP16

</br></br>

---

## SliceFusion

### Function

Tensor slicing operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SkipGram

### Function

The core operation of the Skip-gram model, used for word vector training.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## SmoothL1Loss

### Function

Smooth L1 loss.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## SmoothL1LossGrad

### Function

Computes the gradient of the smooth L1 loss.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Softmax

### Function

Normalization operation.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SoftmaxGrad

### Function

Computes the gradient of Softmax.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Softplus

### Function

Smooth variant of ReLU.

### Supported Hardware Backends

- CPU: FP16, FP32
- Ascend: FP16

</br></br>

---

## SpaceToBatch

### Function

Moves values from height and width dimensions to the depth dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SpaceToBatchND

### Function

Splits blocks of data from spatial dimensions into the batch dimension.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## SpaceToDepth

### Function

Reorganizes spatial data into depth channels.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## SparseToDense

### Function

Converts a sparse representation to a dense tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- GPU (Mali/Adreno): FP16, FP32, Int32

</br></br>

---

## SparseSoftmaxCrossEntropyWithLogits

### Function

Softmax cross-entropy with sparse labels.

### Supported Hardware Backends

- CPU: FP32
- Ascend: FP16

</br></br>

---

## Splice

### Function

Concatenates multiple slices or ranges of the input tensor along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Split

### Function

Splits the input tensor into multiple smaller output tensors along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SplitWithOverlap

### Function

Splits the tensor with overlap.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Sqrt

### Function

Computes the square root element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SqrtGrad

### Function

Computes the gradient of the square root.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Square

### Function

Computes the square element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SquaredDifference

### Function

Computes (A-B)² element-wise.

### Supported Hardware Backends

- CPU: FP16, FP32
- GPU (Mali/Adreno): FP16, FP32

</br></br>

---

## Squeeze

### Function

Removes dimensions of size 1.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- GPU (Mali/Adreno): FP16, FP32, Int32

</br></br>

---

## StridedSlice

### Function

Tensor slicing.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## StridedSliceGrad

### Function

Computes the gradient of the slicing operation.

### Supported Hardware Backends

- CPU: FP16, FP32

</br></br>

---

## Stack

### Function

Stacks multiple tensors along a new axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SubFusion

### Function

Element-wise subtraction.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## SubGrad

### Function

Computes the gradient of subtraction.

### Supported Hardware Backends

- CPU: FP32

</br></br>

---

## Switch

### Function

Selects an output branch based on a boolean condition.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

</br></br>

---

## SwitchLayer

### Function

Selects different sub-network branches for execution in the model.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

</br></br>

---

## TensorListFromTensor

### Function

Converts an ordinary tensor into a tensor list, splitting along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## TensorListGetItem

### Function

Retrieves the tensor at the specified index position from a tensor list.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## TensorListReserve

### Function

Pre-allocates an empty tensor list with the specified element data type and initial capacity.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## TensorListSetItem

### Function

Inserts a tensor at the specified position in a tensor list.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## TensorListStack

### Function

Stacks a tensor list into an ordinary tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## TensorScatterAdd

### Function

Scatter-adds the values of the update tensor to specified positions in the target tensor according to indices.

### Supported Hardware Backends

- CPU: FP32, Int32

</br></br>

---

## TileFusion

### Function

Tiles a given matrix.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool
- Kirin NPU: FP16
- Ascend: FP16

</br></br>

---

## TopKFusion

### Function

Returns the top K elements from the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8
- Ascend: FP16

</br></br>

---

## Transpose

### Function

Tensor transposition.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32
- Ascend: FP16

</br></br>

---

## UniformReal

### Function

Generates a random number tensor that follows a uniform distribution.

### Supported Hardware Backends

- CPU: FP32, Int32

</br></br>

---

## Unique

### Function

Returns the unique values in the input tensor, and optionally returns the indices and counts of those values.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## UnsortedSegmentSum

### Function

Performs segmented summation on a tensor without requiring the segment indices to be ordered.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## Unsqueeze

### Function

Adds a new dimension to the input tensor.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Int8, UInt8, Bool
- Kirin NPU: FP16
- GPU (Mali/Adreno): FP16, FP32, Int32

</br></br>

---

## Unstack

### Function

Splits the tensor into multiple sub-tensors along the specified axis.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

</br></br>

---

## Where

### Function

Element selection.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32, Bool

</br></br>

---

## ZerosLike

### Function

Generates a new tensor with the same shape as the input tensor but filled with zeros.

### Supported Hardware Backends

- CPU: FP16, FP32, Int32

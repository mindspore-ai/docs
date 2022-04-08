# Operators Compile

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/faq/source_en/operators_compile.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: When the `ops.concat` operator is used, the error message `Error:Input and (output + workspace) num should <=192!` is displayed, which indicating that the data volume is large. What can I do?**</font>

A: The `shape` of the `ops.concat` operator is too large. You are advised to set the output to `numpy` when creating an iterator for the `dataset` object. The setting is as follows:

```python
gallaryloader.create_dict_iterator(output_numpy=True)
```

In the post-processing phase (in a non-network calculation process, that is, in a non-`construct` function), `numpy` can be directly used for computation. For example, `numpy.concatenate` is used to replace the `ops.concat` for computation.

<br/>

<font size=3>**Q: In the `construct` function of the static graph mode, how do I remove all negative values contained in a `tensor`?**</font>

A: You are advised to use the `ops.clip_by_value` interface to change all negative numbers to 0 for computation.

<br/>

<font size=3>**Q: What is the function of the `TransData` operator? Can the performance be optimized?**</font>

A: The `TransData` operator is used in the scenario where the data formats (such as NC1HWC0) used by interconnected operators on the network are inconsistent. In this case, the framework automatically inserts the `TransData` operator to convert the data formats into the same format and then performs computation. Huawei Ascend supports 5D format operations, and uses the `transdata` operator to convert data from 4D to 5D to improve performance.

<br/>

<font size=3>**Q: An error occurs when the `Concat` operator concatenates tuples containing multiple tensors. An error occurs when the number of `tensor list` elements entered is greater than or equal to 192. What is a better solution (running in dynamic mode) for `Concat` to concatenate tuples containing multiple Tensors?**</font>

A: The number of tensors to be concatenated at a time cannot exceed 192 according to the bottom-layer specifications of the Ascend operator. You can try to concatenate them twice.

<br/>

<font size=3>**Q: When `Conv2D` is used to define convolution, the `group` parameter is used. Is it necessary to ensure that the value of `group` can be exactly divided by the input and output dimensions? How is the `group` parameter transferred?**</font>

A: The `Conv2d` operator has the following constraint: When the value of `group` is greater than 1, the value must be the same as the number of input and output channels. Do not use `ops.Conv2D`. Currently, this operator does not support a value of `group` that is greater than 1. Currently, only the `nn.Conv2d` API of MindSpore supports group convolution. However, the number of `group` must be the same as the number of input and output channels.

<br/>

<font size=3>**Q: Does MindSpore support matrix transposition?**</font>

A: Yes. For details, see [mindspore.ops.Transpose](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Transpose.html#mindspore.ops.Transpose).

<br/>

<font size=3>**Q: Can MindSpore calculate the variance of any `tensor`?**</font>

A: Currently, MindSpore does not have APIs or operators which can directly calculate the variance of a `tensor`. However, MindSpore has sufficient small operators to support such operations. For details, see [class Moments(Cell)](https://www.mindspore.cn/docs/en/master/_modules/mindspore/nn/layer/math.html#Moments).

<br/>

<font size=3>**Q: Compared with PyTorch, the `nn.Embedding` layer lacks the `padding` operation. Can other operators implement this operation?**</font>

A: In PyTorch, `padding_idx` is used to set the word vector in the `padding_idx` position in the embedding matrix to 0, and the word vector in the `padding_idx` position is not updated during backward propagation.
In MindSpore, you can manually initialize the weight corresponding to the `padding_idx` position of embedding to 0. In addition, the `loss` corresponding to `padding_idx` is filtered out through the `mask` operation during training.

<br/>

<font size=3>**Q: When the `Tile` operator in operations executes `__infer__`, the `value` is `None`. Why is the value lost?**</font>

A: The `multiples input` of the `Tile` operator must be a constant (The value cannot directly or indirectly come from the input of the graph). Otherwise, the `None` data will be obtained during graph composition because the graph input is transferred only during graph execution and the input data cannot be obtained during graph composition. For the detailed imformation, refer to [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

<br/>

<font size=3>**Q: When conv2d is set to (3,10), Tensor [2,2,10,10] and it runs on Ascend on ModelArts, the error message `FM_W+pad_left+pad_right-KW>=strideW` is displayed. However, no error message is displayed when it runs on a CPU. What should I do?**</font>

A: TBE (Tensor Boost Engine) operator is Huawei's self-developed Ascend operator development tool, which is extended on the basis of the TVM framework to develop custom operators. The above problem is the limitation of this TBE operator, and the width of x must be greater than the width of the kernel. The CPU operator does not have this restriction, so no error is reported.

<br/>

<font size=3>**Q: Has MindSpore implemented the anti-pooling operation similar to `nn.MaxUnpool2d`?**</font>

A: Currently, MindSpore does not provide anti-pooling APIs but you can customize the operator to implement the operation. For details, refer to [Customize Operators](https://www.mindspore.cn/docs/programming_guide/en/master/custom_operator.html).

<br/>

<font size=3>**Q: What can I do if the error message `Pynative run op ExpandDims failed` is displayed when the ExpandDims operator is used? The code is as follows:**</font>

```python
context.set_context(mode=context.GRAPH_MODE,device_target='Ascend')
input_tensor=Tensor(np.array([[2,2],[2,2]]),mindspore.float32)
expand_dims=ops.ExpandDims()
output=expand_dims(input_tensor,0)
```

A: The problem is that the Graph mode is selected but the PyNative mode is used. As a result, an error is reported. MindSpore supports the following running modes which are optimized in terms of debugging or running:

- PyNative mode: dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.
- Graph mode: static graph mode or graph mode. In this mode, the neural network model is compiled into an entire graph and then delivered for execution. This mode uses technologies such as graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.

<br/>

<font size=3>**Q: Ascend backend error: `AI CORE` and `AI CPU` can not find a valid `kernel info` when the Kernel Select Failed, how to locate?**</font>

A: The `Ascend` backend operators can be divided into AI CORE operators and AI CPU operators. Some operators are supported by AI CORE, some operators are supported by AI CPU, and some operators are supported by AI CORE and AI CPU at the same time. According to the error message:

1. If the `AI CORE` operator's candidates list is empty, it may be that all operator information failed to pass the verification in the `check support` stage. You can search the keyword `CheckSupport` in the log to find the reason for the failure. Modify the shape or data type according to the specific information, or ask the developer to further locate the problem.
2. If the `AI CPU` candidate operator information is not empty, or the candidate operator information of `AI CORE` and `AI CPU` are both not empty, it may be that the given input data type was not in the candidate list and was filtered out in the selection stage. Try to modify the input data type of the operator according to the candidate list.

You can select a proper mode and writing method to complete the training by referring to the [official website tutorial](https://www.mindspore.cn/docs/programming_guide/en/master/debug_in_pynative_mode.html).


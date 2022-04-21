# FAQ

<a href="https://gitee.com/mindspore/docs/blob/master/docs/graphlearning/docs/source_en/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What should I do if the error message `OSError: could not get source code` is displayed during the execution of the GNNCell command?**</font>

A: MindSpore Graph Learning parses vertex-centeric programming code through source-to-source translation. The inspect function is called to obtain source code of the translation module. Model definition code needs to be placed in a Python file. Otherwise, an error message is displayed, indicating that the source code cannot be found.

<br/>

<font size=3>**Q: What should I do if the error message `AttributeError: None of backend from {mindspore} is identified. Backend must be imported as a global variable.` is displayed during the execution of GNNCell?**</font>

A: MindSpore Graph Learning parses vertex-centeric programming code through source-to-source translation. In the GNNCell definition file, the network execution backend is obtained based on global variables. You need to import MindSpore in the header of the GNNCell definition file. Otherwise, an error is reported, indicating that the backend cannot be found.

<br/>

<font size=3>**Q: What should I do if the error message `TypeInferenceError: Line 6: Built-in agg func "avg" only takes expr of EDGE or SRC type. Got None.` is displayed when the graph aggregation APIs 'sum, avg, max, and min' is called?**</font>

A: The aggregate API provided by MindSpore Graph Learning performs operations on graph nodes. During source-to-source translation, the system checks whether the input of the aggregate API is an edge or a node in the graph. If not, an error is reported, indicating that the required input type cannot be found.

<br/>

<font size=3>**Q: What should I do if the error message `RuntimeError: The 'mul' operation does not support the type.` is displayed when I call the graph API 'dot'?**</font>

A: The dot API provided by MindSpore Graph Learning is used to perform the dot multiplication operation on graph nodes. The backend includes feature multiplication and aggregation. The frontend translation process does not involve build and cannot determine the input data type. The input type must meet the type requirements of the backend mul operator. Otherwise, an error message is displayed, indicating that the type is not supported.

<br/>

<font size=3>**Q: What should I do if the error message `TypeError: For 'tensor getitem', the types only support 'Slice', 'Ellipsis', 'None', 'Tensor', 'int', 'List', 'Tuple', 'bool', but got String.` is displayed when I call the graph API 'topk_xxx'?**</font>

A: The topk_xxx API provided by MindSpore Graph Learning is used to obtain k nodes or edges based on node or edge feature sorting. The backend includes three steps: obtaining node or edge features, sorting, and slicing k nodes or edges. The frontend translation process does not involve build and cannot determine the input data type. The input type must meet the sorting dimension sortby and value range k of the sort and slice operators. Otherwise, an error is reported, indicating that the type is not supported.

<br/>

<font size=3>**Q: What should I do if the error message `TypeError: For 'Cell', the function construct need 5 positional argument, but got 2.'` is displayed when a non-GraphField instance or equivalent tensors are passed to the input graph of the construct function?**</font>

A: The GNNCell class provided by MindSpore Graph Learning is a base class for writing vertex-centeric programming GNN models. It must contain the graph class as the last input parameter. The translated input is four tensor parameters, which are src_idx, dst_idx, n_nodes, and n_edges. If only a non-GraphField instance or four equivalent tensors are passed, an error message is displayed, indicating that the input parameter is incorrect.

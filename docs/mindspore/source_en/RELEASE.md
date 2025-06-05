# MindSpore Release Notes

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/mindspore/source_en/RELEASE.md)

## MindSpore 2.6.0 Release Notes

### Major Features and Improvements

#### Dataset

- [STABLE] The sharding sampling behavior of the [MindDataset](https://www.mindspore.cn/docs/en/r2.6.0/api_python/dataset/mindspore.dataset.MindDataset.html) interface has been changed from block-based sampling (Data sharding strategy 2 in the link) to interval sampling (Data sharding strategy 1 in the link). Users can control whether to switch back to block-based sampling by setting the MS_DEV_MINDRECORD_SHARD_BLOCK environment variable.
- [STABLE] GeneratorDataset supports spawn to start multiprocessing, and supports the use of Ascend back-end data augmentation methods in multiprocessing. Users can set [mindspore.dataset.config.set_multiprocessing_start_method("spawn")](https://www.mindspore.cn/docs/en/r2.6.0/api_python/dataset/mindspore.dataset.config.set_multiprocessing_start_method.html) to enable multiprocessing in spawn mode.
- [STABLE] The `shuffle` parameter in [MindDataset](https://www.mindspore.cn/docs/en/r2.6.0/api_python/dataset/mindspore.dataset.MindDataset.html) supports the `Shuffle.ADAPTIVE`option, which adaptively adjusts the shuffle sample count strategy based on the number of samples to reduce training memory overhead and lower the risk of OOM. If global shuffle is desired, users can specify `Shuffle.GLOBAL`, but they must ensure sufficient machine memory.

#### Ascend

- [STABLE] In MindSpore's dynamic graph mode, the AscendC custom operators integrated by the [ops.Custom](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.Custom.html) primitive support multiple output types, and `ops.Custom` supports type inference on the C++ side.
- [BETA] In MindSpore's dynamic graph mode, added `CustomOpBuilder` to support online building and loading of custom operators.
- [STABLE] When using the O1 compilation option, users can control the scope of graph and computation fusion optimization. Users can enable or disable specific fusion patterns by setting the environment variable MS_DEV_GRAPH_KERNEL_FLAGS with options such as enable_fusion_pattern_only or disable_fusion_pattern. Additionally, it supports reading configuration from a file via the --path=example.json option.
- [STABLE] Support users to set the aclop operator cache information aging configuration and error message reporting mode configuration through the [mindspore.device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/en/r2.6.0/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html) interface.
- [STABLE] GE backend only supports whole graph sinking and lazy inline subgraph sinking, while other scenarios are no longer supported.
- [BETA] In MindSpore's static graph O0/O1 mode, `mindpore.nn.Cell` adds the new interface `offload` and the attribute `backward_prefetch`. Users can use this interface through [Cell.offload(backward_prefetch)](https://www.mindspore.cn/docs/en/r2.6.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.offload) to offload activations within a specific `Cell` class from the device side to the host side during the forward training phase, and prefetch activations from the host side to the device side during the backward training phase.

#### Parallel

- [STABLE] Parallel pdb debugging, dynamic and static graph mode are supported. dynamic graph mode is recommended.
- [STABLE] New API [mindspore.communication.get_comm_name](https://www.mindspore.cn/docs/en/r2.6.0/api_python/communication/mindspore.communication.get_comm_name.html), which allows users to query the name of the underlying communicator of the HCCL collection communication library.
- [STABLE] Added [AutoParallel](https://www.mindspore.cn/docs/en/r2.6.0/api_python/parallel/mindspore.parallel.auto_parallel.AutoParallel.html) API to support parallel configuration of individual networks, solving the problem of excessive scope of parallel configuration.
- [STABLE] SeqPipe now supports two new scheduling methods, seqvpp and seqsmartvpp, significantly reducing the memory cost in scenarios where SeqPipe is combined with VPP.
- [STABLE] Static graph now supports zero2/zero3 level memory optimization, reducing the memory cost for models that require pure data parallel (DP) training.
- [STABLE] Static graph now supports 1b1f compute and communication overlapping in pipeline parallelism conditions, enhancing the performance of pipeline parallelism.
- [STABLE] Static graphs support grad model parallel communication overlap with dw computation under tensor model parallelism and expert model parallelism, improving model training performance.
- [STABLE] Static graph auto-parallel strategy propagation mode is updated to prioritize the layout propagation to improve the accuracy.
- [STABLE] Static graph auto-parallel support using [mindspore.parallel.shard](https://www.mindspore.cn/docs/en/r2.6.0/api_python/parallel/mindspore.parallel.shard.html) interface to configure strategies for mint operators, optimized for multi-input operators.
- [STABLE] For LLM reinforcement learning， now we support DP/MP/PP for training and inferenceing phase.
- [STABLE] MindSpore supports users to query whether the distributed module is available and whether the communication module is initialized. Users can query whether the distributed module is available through the [mint.distributed.is_available](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.distributed.is_available.html) interface, and query whether the communication module is initialized through the [mint.distributed.is_initialized](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.distributed.is_initialized.html) interface.
- [STABLE] MindSpore static graph mode supports the `AlltoAllV` forward and reverse operators. Users can use this operator through the [ops.AlltoAllV](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.AlltoAllV.html) interface.
- [STABLE] Support CPU operators [mindspore.mint.distributed.allreduce](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.distributed.all_reduce.html#mindspore.mint.distributed.all_reduce), [mindspore.mint.distributed.barrier](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.distributed.barrier.html#mindspore.mint.distributed.barrier), [mindspore.mint.distributed.send](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.distributed.send.html#mindspore.mint.distributed.send), and [mindspore.mint.distributed.recv](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.distributed.recv.html#mindspore.mint.distributed.recv), and the users can use the corresponding aggregate communication operator functions through these interfaces.

#### Inference

- [STABLE] Support full-precision inference with BFloat16 and quantized inference with W8A8 for DeepSeek-V3/R1. Add or optimize 12 fusion operators including RmsNormQuant, MatMul+Sigmoid+Add, and Transpose+BatchMatMul+Transpose to enhance the inference performance of DeepSeek-V3/R1.
- [BETA] Support deploying inference services of DeepSeek-V3/R1 using MindIE and MindSpore Transformers large model development suite.
- [STABLE] Optimize the process of loading safetensors and realize on-demand initialization of GE, which reduces both memory usage and startup time when deploying inference services using MindIE and MindSpore Transformers large model suite.
- [BETA] Support deploying inference services of DeepSeek-V3/R1 and Qwen2.5 using the [vLLM-MindSpore](https://gitee.com/mindspore/vllm-mindspore) plugin and vLLM v0.6.6.post1.

#### Profiler

- [STABLE] The MindSpore framework supports obtaining communication domain parallel strategy information, which can be visualized to improve performance troubleshooting efficiency in cluster scenarios.
- [STABLE] MindSpore Profiler dynamic profiling supports lightweight instrumentation, allowing users to dynamically enable lightweight tracing and view performance data in real time.
- [STABLE] MindSpore Profiler's lightweight instrumentation capability has been enhanced, supporting key phases such as dataloader and save checkpoint with lightweight tracing information.
- [STABLE] Profiler supports viewing memory_access related aicore metric information.
- [STABLE] MindSpore Profiler supports [mindspore.profiler.profile](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.profiler.profile.html) and [_ExperimentalConfig](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.profiler._ExperimentalConfig.html), as well as the [tensorboard_trace_handler](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html) parameter, improving tool usability.
- [STABLE] MindSpore Profiler dynamic profiling now supports memory data collection, allowing users to dynamically enable memory data gathering to enhance tool usability.

#### Compiler

- [BETA] The graph mode supports the inplace and view operator forward expression capabilities.
- [BETA] Add new operator primitive [mindspore.ops.Morph](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.Morph.html) in GRAPH mode, enabling encapsulation of user-defined function as operator primitive [mindspore.ops.Morph](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.Morph.html), facilitating encapsulation of irregular collective communication operations (such as [mindspore.ops.AlltoAllV](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.AlltoAllV.html)) for distributed auto-parallel training scenarios.

### API Change

#### New APIs & Enhanced APIs

- [DEMO] [mindspore.mint](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore.mint.html) API provides more functional, nn interfaces. The mint interface is currently an experimental interface and performs better than ops in `jit_level="O0"` and pynative mode. Currently, the graph sinking mode and CPU/GPU backend are not supported, and it will be gradually improved in the future.

  | mindspore.mint                  |
  | :------------------------------ |
  | mindspore.mint.reshape          |
  | mindspore.mint.triangular_solve |
  | mindspore.mint.index_add        |
  | mindspore.mint.logaddexp2       |
  | mindspore.mint.diag             |

  | mindspore.mint.nn              |
  | :----------------------------- |
  | mindspore.mint.nn.Sigmoid      |
  | mindspore.mint.nn.Conv2d       |
  | mindspore.mint.nn.PixelShuffle |

  | mindspore.mint.nn.functional                     |
  | :----------------------------------------------- |
  | mindspore.mint.nn.functional.adaptive_avg_pool3d |
  | mindspore.mint.nn.functional.conv2d              |
  | mindspore.mint.nn.functional.avg_pool3d          |
  | mindspore.mint.nn.functional.elu_                |
  | mindspore.mint.nn.functional.pixel_shuffle       |

  | others                   |
  | ------------------------ |
  | mindspore.mint.optim.SGD |
  | mindspore.mint.linalg.qr |

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore.mint.html) API also provides some new stable interfaces. Besides, some demo interfaces are changed into stable ones.

  | mindspore.mint           |
  | :----------------------- |
  | mindspore.mint.full_like |
  | mindspore.mint.log2      |
  | mindspore.mint.isneginf  |

  | mindspore.mint.nn           |
  | :-------------------------- |
  | mindspore.mint.nn.GLU       |
  | mindspore.mint.nn.KLDivLoss |

  | mindspore.mint.nn.functional        |
  | :---------------------------------- |
  | mindspore.mint.nn.functional.glu    |
  | mindspore.mint.nn.functional.kl_div |

  | mindspore.Tensor          |
  | :------------------------ |
  | mindspore.Tensor.isneginf |
  | mindspore.Tensor.log2     |

- [DEMO] [mindspore.Tensor](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) API provides more Tensor methods. Currently, these Tensor methods are experimental interfaces and currently does not support the graph sink mode and CPU, GPU backend, and they will be gradually improved in the future. Details can be found in [API list](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) in official website.
- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore.ops.html) provides two inference API [mindspore.ops.moe_token_permute](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.moe_token_permute.html#mindspore.ops.moe_token_permute) and [mindspore.ops.moe_token_unpermute](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.moe_token_unpermute.html#mindspore.ops.moe_token_unpermute). Currently, only Ascend backend is supported.
- [STABLE] [mindspore.mint.nn.functional.gelu](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.nn.functional.gelu.html) and [mindspore.mint.nn.GeLU](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mint/mindspore.mint.nn.GELU.html) now support input argument "approximate".
- [STABLE] Added the offline parsing interface [mindspore.profiler.profiler.analyse](https://gitee.com/link?target=https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.profiler.profiler.analyse.html).

#### Backwards Incompatible Change

- For [mindspore.ops.Xlogy](https://www.mindspore.cn/docs/en/r2.6.0/api_python/ops/mindspore.ops.Xlogy.html), the arguments `input` and `other` no longer support non-tensor input. [(!81625)
  ](https://gitee.com/mindspore/mindspore/pulls/81625)

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  ops.Xlogy(input [Tensor, numbers.Number, bool],
            other [Tensor, numbers.Number, bool])
  </td>
  <td><pre>
  ops.Xlogy(input [Tensor], other [Tensor])
  </td>
  </tr>
  </table>

- `&` operator no longer supports the input Tensor with data type of uint32/uint64 on Ascend backend in PyNative mode.  `^` operator no longer supports the  input Tensor with data type of uint16/uint32/uint64 on Ascend backend in PyNative mode. `|` operator no longer supports the input Tensor with data type of uint16/uint32/uint64 on Ascend backend in PyNative mode at the scene of `tensor | scalar`. [(!82054)](https://gitee.com/mindspore/mindspore/pulls/82054)
- `%` operator no longer supports the input Tensor with data type of uint16/uint32/uint64 on CPU and GPU backend. [(!83055)](https://gitee.com/mindspore/mindspore/pulls/83055)
- [mindspore.jit](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.jit.html) interface parameter change。[(!80248)](https://gitee.com/mindspore/mindspore/pulls/80248)

  The name of parameter `fn` is changed to `function` .

  Remove parameter `mode` , `input_signature` , `hash_args` , `jit_config` and `compile_once` .

  Add parameter `capture_mode` to set how to compile to MindSpore graph.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(mode="PIJit")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(capture_mode="bytecode")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `jit_level` to set the level of compilation optimization.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit, JitConfig
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_config=JitConfig(jit_level="O0"))
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_level="O0")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `dynamic` to set whether dynamic shape compilation should be performed.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(dynamic=1)
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `fullgraph` to set whether to capture the entire function into graph.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit, JitConfig
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_config=JitConfig(jit_syntax_level="STRICT"))
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(fullgraph=True)
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `backend` to set the compilation backend to be used.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(backend="ms_backend")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

  Add parameter `options` to set the dictionary of options to pass to the compilation backend.

  <table>
    <tr>
    <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
    </tr>
    <tr>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit, JitConfig
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(jit_config=JitConfig(infer_boost="on"))
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    <td><pre>
    >>> import numpy as np
    >>> from mindspore import Tensor, jit
    >>>
    >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>>
    >>> @jit(infer_boost="on")
    ... def tensor_add_with_dec(x, y):
    ...     z = x + y
    ...     return z
    ...
    >>> out = tensor_add_with_dec(x, y)
    </pre>
    </td>
    </tr>
  </table>

- The `mindspore.profiler.tensor_board_trace_handler` interface change.

  The `mindspore.profiler.tensor_board_trace_handler` interface is now renamed to [mindspore.profiler.tensorboard_trace_handler](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html).

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  >>> from mindspore.profiler import tensor_board_trace_handler
  </pre>
  </td>
  <td><pre>
  >>> from mindspore.profiler import tensorboard_trace_handler
  </pre>
  </td>
  </tr>
  </table>

- The  `mindspore.set_context` interface change。

  The `exception_dump` field in the `ascend_config` parameter was changed to the `"dump"` field in [device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/en/r2.6.0/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html).

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  >>> import mindspore as ms
  >>> ms.set_context(
  ...     ascend_config = {"exception_dump": "2"}
  ...     )
  </pre>
  </td>
  <td><pre>
  >>> import mindspore as ms
  >>> ms.device_context.ascend.op_debug.aclinit_config(
  ...     {"dump": {"dump_scene": "lite_exception"}}
  ...     )
  </pre>
  </td>
  </tr>
  </table>

- The printing content of `mindspore.Tensor` change。

  The original Tensor prints only the value, while the new Tensor prints key information such as shape and dtype.

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  >>> import mindspore as ms
  >>> tensor = ms.Tensor([1,1,1], dtype=ms.float32)
  >>> print(tensor)
  [1. 1. 1.]
  </pre>
  </td>
  <td><pre>
  >>> import mindspore as ms
  >>> tensor = ms.Tensor([1,1,1], dtype=ms.float32)
  >>> print(tensor)
  Tensor(shape=[3],
         dtype=Float32,
         value= [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00])
  </pre>
  </td>
  </tr>
  </table>

- In graph mode, Ascend backend, when jit_level is O2, the Dump interface changes.

  In the graph Ascend backend jit_level O2 scenario, the environment variables `MINDSPORE_DUMP_CONFIG` and `ENABLE_MS_GE_DUMP` have been deprecated, and the dump-related functions have been migrated to the msprobe tool. For more details, please refer to [msprobe Tool MindSpore Scene Accuracy Data Collection Guide](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md).

### Contributors

amyMaYun,Ava,baishanyang,br_fix_save_strategy_ckpt,caifubi,caoxubo,cccc1111,ccsszz,chaijinwei,chaiyouheng,changzherui,chengbin,chopupu,chujinjin,congcong,dairenjie,DavidFFFan,DeshiChen,dingjinshan,fary86,fengqiang,fengyixing,ffmh,fuhouyu,Gallium,gaoshuanglong,gaoyong10,geyuhong,guoyq16,guoyuzhe,GuoZhibin,guozhijian,gupengcheng0401,hangq,Hanshize,haozhang,hedongdong,hhz886,HighCloud,horcham,huangbingjian,huangxiang360729,huangzhichao2023,huangzhuo,huangziling,huda,Huilan Li,hujiahui8,huoxinyou,jiangchao_j,jiangchenglin3,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,JoeyLin,jshawjc,kairui_kou,kakyo82,kisnwang,leida,lianghongrui,LiangZhibo,LiaoTao_Wave,lichen,limingqi107,LiNuohang,linux,litingyu,liubuyu,liuchuting,liuluobin,liuyanwei,LLLRT,looop5,luochao60,luojianing,luoxuewei,luoyang,lyk,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mylinchi,NaCN,panzhihui,pengqi,PingqiLi,pipecat,qiuyufeng,qiuzhongya,qiwenlun,r1chardf1d0,rachel0858,rainyhorse,Rudy_tan,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shiro-zzz,shuqian0,stavewu,TAJh,tanghuikang,tangmengcheng,tongdabao,TuDouNi,VectorSL,wang_ziqi,wangjie,wangliaohui97,wangpingan,wangyibo,weiyang,wja,wudongkun,wujueying,wuweikang,wwwbby,xfan233,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,xiedejin1,XinDu,xuxinglei,xuzhen,xuzixiang,yang guodong,yangben,yanghaoran,yangruoqi713,yangzhenzhang,yanx,Yanzhi_YI,yao_yf,yide12,yihangchen,YijieChen,yonibaehr,Youmi,yuanqi,yuchaojie,yuezenglin,Yuheng Wang,YuJianfeng,YukioZzz,ZeyuHan,zhaiyukun,Zhang QR,zhangbuxue,zhangdanyang,zhangshucheng,zhangyinxia,ZhangZGC,zhangzhen,zhengzuohe,zhouyaqiang0,zichun_ye,zlq2020,zong_shuai,ZPaC,zyli2020,舛扬,范吉斌,冯一航,胡犇,胡彬,宦晓玲,简云超,李栋,李良灿,李林杰,李寅杰3,刘思铭,刘勇琪,刘子涵,梅飞要,任新,十一雷,孙昊辰,王泓皓,王禹程,王振邦,熊攀,俞涵,虞良斌,云骑士,张栩浩,赵文璇,周一航

# MindSpore Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/mindspore/source_zh_cn/RELEASE.md)

## MindSpore 2.6.0 Release Notes

### 主要特性及增强

#### Dataset

- [STABLE] [MindDataset](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/dataset/mindspore.dataset.MindDataset.html)接口分片采样行为由原来的按块采样（链接中的数据分片的策略2）变更为间隔采样（链接中的数据分片的策略1），用户可以通过 MS_DEV_MINDRECORD_SHARD_BY_BLOCK 环境变量，控制是否切换回按块采样。
- [STABLE] GeneratorDataset 支持 spawn 方式启动多进程，支持在多进程时，使用Ascend后端的数据增强方法。用户可以设置 [mindspore.dataset.config.set_multiprocessing_start_method("spawn")](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/dataset/mindspore.dataset.config.set_multiprocessing_start_method.html) 以 spawn 的方式启动多进程。
- [STABLE] [MindDataset](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/dataset/mindspore.dataset.MindDataset.html) 的 `shuffle` 参数新增了 `Shuffle.ADAPTIVE` 行为，根据样本数量自适应调整 shuffle 样本数量的策略以降低训练内存开销，减少 OOM 风险。若期望强制采用全局 shuffle，可以指定为 `Shuffle.GLOBAL`，用户需确保机器内存足够。

#### Ascend

- [STABLE] 动态图模式场景，[ops.Custom](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.Custom.html) 原语接入Ascend C自定义算子，支持多输出类型，`ops.Custom`支持C++侧infer type。
- [BETA] 动态图模式场景，新增CustomOpBuilder支持在线编译和加载自定义算子。
- [STABLE] 使用O1编译选项时，支持用户控制图算融合优化范围，用户通过环境变量MS_DEV_GRAPH_KERNEL_FLAGS的enable_fusion_pattern_only/disable_fusion_pattern选项，控制打开或者关闭对应融合pattern，同时支持通过--path=example.json方式读取文件配置。
- [STABLE] 支持通过 [mindspore.device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html) 接口，设置aclop算子缓存信息老化配置和错误信息上报模式配置。
- [STABLE] GE后端仅支持整图下沉和lazy inline子图下沉，其他场景不再支持。
- [BETA] 静态图O0/O1模式场景，`mindspore.nn.Cell`基类新增offload接口与backward_prefetch接口属性，用户可通过[Cell.offload(backward_prefetch)](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.offload) 使用该接口，在训练正向阶段，将特定`Cell`类内的激活值从device侧卸载至host侧，并在训练反向阶段，将激活值从host侧提前预取回device侧。

#### Parallel

- [STABLE] 分布式pdb调试，支持动态图和静态图，更推荐使用动态图。
- [STABLE] 新增接口[mindspore.communication.get_comm_name](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/communication/mindspore.communication.get_comm_name.html)，用户可以通过该接口查询HCCL集合通信库底层通信器名称。
- [STABLE] 新增 [AutoParallel](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/parallel/mindspore.parallel.auto_parallel.AutoParallel.html) 接口，支持对单个网络进行并行配置，解决并行配置作用域过大的问题。
- [STABLE] seqpipe新增支持两种调度方式seqvpp、seqsmartvpp，显著降低seqpipe结合vpp场景下的显存开销。
- [STABLE] 静态图模式场景，支持zero2/zero3级别的内存优化，降低有纯dp训练需求的模型的显存开销。
- [STABLE] 静态图模式场景，支持流水线并行下的1b1f通信掩盖，提升流水线并行性能。
- [STABLE] 静态图模式场景，支持张量模型并行和专家模型并行下的反向通信掩盖，提升模型训练性能。
- [STABLE] 静态图模式场景，自动并行策略传播模式更新为优先传播算子的Layout策略，提高策略传播准确性。
- [STABLE] 静态图模式场景，自动并行支持使用 [mindspore.parallel.shard](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/parallel/mindspore.parallel.shard.html) 接口为mint算子配置策略，优化了多输入算子的策略。
- [STABLE]支持强化学习场景中，DP/MP/PP 多维混合并行模式下的训推权重在线权重重排。
- [STABLE] 支持用户查询分布式模块是否可用和通信模块是否初始化功能，用户可以通过 [mint.distributed.is_available](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.distributed.is_available.html) 接口查询分布式模块是否可用，以及通过 [mint.distributed.is_initialized](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.distributed.is_initialized.html) 接口查询通信模块是否初始化。
- [STABLE] 静态图模式场景，支持 `AlltoAllV` 正反向算子，用户可通过 [ops.AlltoAllV](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.AlltoAllV.html) 接口使用该算子。
- [STABLE] 支持CPU通信接口[mindspore.mint.distributed.allreduce](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.distributed.all_reduce.html#mindspore.mint.distributed.all_reduce)、[mindspore.mint.distributed.barrier](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.distributed.barrier.html#mindspore.mint.distributed.barrier)、[mindspore.mint.distributed.send](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.distributed.send.html#mindspore.mint.distributed.send)、[mindspore.mint.distributed.recv](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.distributed.recv.html#mindspore.mint.distributed.recv)，用户可通过这些接口使用对应的集合通信算子功能。

#### Inference

- [STABLE] 支持DeepSeek-V3/R1大模型的BFloat16全精度推理和W8A8量化推理，并为提升其推理性能开发或优化了RmsNormQuant、MatMul+Sigmoid+Add、Transpose+BatchMatMul+Transpose等12个融合算子。
- [BETA] 支持使用MindIE和MindSpore Transformers大模型套件，服务化部署DeepSeek-V3/R1。
- [STABLE] 优化了使用MindIE和MindSpore Transformers大模型套件进行推理服务部署时，加载safetensors的过程，实现了GE按需初始化，分别降低了内存占用量和启动耗时。
- [BETA] 支持使用[vLLM-MindSpore](https://gitee.com/mindspore/vllm-mindspore)插件和vLLM v0.6.6.post1，服务化部署DeepSeek-V3/R1、Qwen2.5大模型。

#### profiler

- [STABLE] 支持获取通信域并行策略信息，并行策略信息支持可视化显示，提升集群场景下性能定位效率。
- [STABLE] 动态profiling支持轻量化打点，用户可动态开启轻量化打点，实时查看性能数据。
- [STABLE] Profiler轻量化打点能力增强，支持dataloader、save checkpoint等关键阶段轻量化打点信息。
- [STABLE] Profiler支持查看memory_access相关aicore metric信息。
- [STABLE] Profiler支持[mindspore.profiler.profile](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.profiler.profile.html)、[_ExperimentalConfig](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.profiler._ExperimentalConfig.html) 和[tensorboard_trace_handler](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html)，提升工具易用性。
- [STABLE] 动态profiling支持内存采集，用户可动态开启内存数据采集，提升工具易用性。

#### Compiler

- [BETA] 图模式支持inplace和view算子正向表达能力。

### API 变更

#### 新增API

- [DEMO] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore.mint.html) API新增了大量的functional、nn接口。mint接口当前是实验性接口，在图编译模式为O0/O1和PyNative模式下性能比ops更优。当前暂不支持O2编译模式(图下沉)及CPU、GPU后端，后续会逐步完善。

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

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore.mint.html) API 也提供了一些新增的stable接口。此外, 一些demo的接口也转为了stable。

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

- [DEMO] [mindspore.Tensor](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) API新增了大量的Tensor方法的接口。当前仍属于实验性接口，当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。详细见[官网接口列表](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor)。
- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore.ops.html) API新增[mindspore.ops.moe_token_permute](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.moe_token_permute.html#mindspore.ops.moe_token_permute) 和 [mindspore.ops.moe_token_unpermute](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.moe_token_unpermute.html#mindspore.ops.moe_token_unpermute)两个推理算子接口，当前仅支持Ascend后端。
- [STABLE] [mindspore.mint.nn.functional.gelu](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.nn.functional.gelu.html) 和 [mindspore.mint.nn.GeLU](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mint/mindspore.mint.nn.GELU.html) 新增支持了入参 "approximate"。
- [STABLE] 新增离线解析接口[mindspore.profiler.profiler.analyse](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.profiler.profiler.analyse.html)。

#### 非兼容性接口变更

- [mindspore.ops.Xlogy](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.Xlogy.html) 接口入参input和other移除了对非Tensor输入的支持！[(!81625)](https://gitee.com/mindspore/mindspore/pulls/81625)

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  ops.Xlogy(input [Tensor, numbers.Number, bool],
            other [Tensor, numbers.Number, bool])
  </pre>
  </td>
  <td><pre>
  ops.Xlogy(input [Tensor],
            other [Tensor])
  </pre>
  </td>
  </tr>
  </table>

- `&`运算符在Ascend后端PyNative模式下不再支持uint32、uint64类型的Tensor输入，`^`运算符在Ascend后端PyNative模式下不再支持uint16、uint32、uint64类型的Tensor输入，`|`运算符在Ascend后端PyNative模式 `tensor | scalar`的场景下不再支持uint16、uint32、uint64类型的Tensor输入。[(!81625)](https://gitee.com/mindspore/mindspore/pulls/81625)
- `%`运算符在CPU和GPU后端不再支持uint16、uint32、uint64类型的Tensor输入。[(!81625)](https://gitee.com/mindspore/mindspore/pulls/81625)
- [mindspore.jit](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.jit.html) 接口参数变更。[(!80248)](https://gitee.com/mindspore/mindspore/pulls/80248)

  参数 `fn` 名称变更为 `function` 。

  移除参数 `mode` 、 `input_signature` 、 `hash_args` 、 `jit_config` 和 `compile_once` 。

  新增参数 `capture_mode` ，设置编译成MindSpore图的方式。

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

  新增参数 `jit_level` ，设置编译优化的级别。

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

  新增参数 `dynamic` ，设置是否需要进行动态shape编译。

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

  新增参数 `fullgraph` ，设置是否捕获整个函数来编译成图。

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

  新增参数 `backend` ，设置使用的编译后端。

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

  新增参数 `options` ，设置传给编译后端的选项字典。

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

- `mindspore.profiler.tensor_board_trace_handler` 接口变更。

  `mindspore.profiler.tensor_board_trace_handler`接口变更为 [mindspore.profiler.tensorboard_trace_handler](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore/mindspore.profiler.tensorboard_trace_handler.html)。

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

- `mindspore.set_context`接口变更。

  参数 `ascend_config` 中的 `exception_dump`字段变更为 [device_context.ascend.op_debug.aclinit_config](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/device_context/mindspore.device_context.ascend.op_debug.aclinit_config.html) 中的 `"dump"`字段。

  <table>
  <tr>
  <td style="text-align:center"> 2.5.0 </td> <td style="text-align:center"> 2.6.0 </td>
  </tr>
  <tr>
  <td><pre>
  >>> import mindspore as ms
  >>> ms.set_context(ascend_config = {"exception_dump": "2"})
  </pre>
  </td>
  <td><pre>
  >>> import mindspore as ms
  >>> ms.device_context.ascend.op_debug.aclinit_config({"dump": {"dump_scene": "lite_exception"}})
  </pre>
  </td>
  </tr>
  </table>

- `mindspore.Tensor`打印内容变更。

  原有Tensor打印内容，只打印值，新Tensor打印内容包含shape和dtype等Tensor关键信息。

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
  Tensor(shape=[3], dtype=Float32, value= [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00])
  </pre>
  </td>
  </tr>
  </table>

- 静态图模式，Ascend后端，jit_level为O2模式下Dump接口变更。

  在静态图Ascend 后端 jit_level为O2场景下，环境变量 `MINDSPORE_DUMP_CONFIG`和 `ENABLE_MS_GE_DUMP`已废弃，Dump相关功能已迁移到 msprobe 工具，更多详情请查看[《msprobe 工具 MindSpore 场景精度数据采集指南》](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/06.data_dump_MindSpore.md)。

### 贡献者

amyMaYun,Ava,baishanyang,br_fix_save_strategy_ckpt,caifubi,caoxubo,cccc1111,ccsszz,chaijinwei,chaiyouheng,changzherui,chengbin,chopupu,chujinjin,congcong,dairenjie,DavidFFFan,DeshiChen,dingjinshan,fary86,fengqiang,fengyixing,ffmh,fuhouyu,Gallium,gaoshuanglong,gaoyong10,geyuhong,guoyq16,guoyuzhe,GuoZhibin,guozhijian,gupengcheng0401,hangq,Hanshize,haozhang,hedongdong,hhz886,HighCloud,horcham,huangbingjian,huangxiang360729,huangzhichao2023,huangzhuo,huangziling,huda,Huilan Li,hujiahui8,huoxinyou,jiangchao_j,jiangchenglin3,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,JoeyLin,jshawjc,kairui_kou,kakyo82,kisnwang,leida,lianghongrui,LiangZhibo,LiaoTao_Wave,lichen,limingqi107,LiNuohang,linux,litingyu,liubuyu,liuchuting,liuluobin,liuyanwei,LLLRT,looop5,luochao60,luojianing,luoxuewei,luoyang,lyk,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mylinchi,NaCN,panzhihui,pengqi,PingqiLi,pipecat,qiuyufeng,qiuzhongya,qiwenlun,r1chardf1d0,rachel0858,rainyhorse,Rudy_tan,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shiro-zzz,shuqian0,stavewu,TAJh,tanghuikang,tangmengcheng,tongdabao,TuDouNi,VectorSL,wang_ziqi,wangjie,wangliaohui97,wangpingan,wangyibo,weiyang,wja,wudongkun,wujueying,wuweikang,wwwbby,xfan233,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,xiedejin1,XinDu,xuxinglei,xuzhen,xuzixiang,yang guodong,yangben,yanghaoran,yangruoqi713,yangzhenzhang,yanx,Yanzhi_YI,yao_yf,yide12,yihangchen,YijieChen,yonibaehr,Youmi,yuanqi,yuchaojie,yuezenglin,Yuheng Wang,YuJianfeng,YukioZzz,ZeyuHan,zhaiyukun,Zhang QR,zhangbuxue,zhangdanyang,zhangshucheng,zhangyinxia,ZhangZGC,zhangzhen,zhengzuohe,zhouyaqiang0,zichun_ye,zlq2020,zong_shuai,ZPaC,zyli2020,舛扬,范吉斌,冯一航,胡犇,胡彬,宦晓玲,简云超,李栋,李良灿,李林杰,李寅杰3,刘思铭,刘勇琪,刘子涵,梅飞要,任新,十一雷,孙昊辰,王泓皓,王禹程,王振邦,熊攀,俞涵,虞良斌,云骑士,张栩浩,赵文璇,周一航

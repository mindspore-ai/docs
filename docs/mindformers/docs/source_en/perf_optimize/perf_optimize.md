# Large Model Performance Optimization Guide

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/perf_optimize/perf_optimize.md)

## Performance Optimization Descritions

This document mainly introduces the performance tuning of large language model, detailed introduction to the basic theoretical knowledge related to performance tuning, analysis ideas and related tools to use guidance, as well as case sharing.

Performance generally includes in terms of model training performance, with the time required to complete a single end-to-end training session as a reference metric, given a specified model and input data. End-to-end refers to the process of completing a single-step training of an AI model, and the time is mainly composed of the following components:

* Data loading time: it refers to the time for the model to load the training data, weights, and other data, including reading the data from the hardware storage device into the CPU, preprocessing the data in the CPU, and carrying the CPU data to the NPU. For some models that need to be sliced onto several NPUs, the data loading also includes the time to broadcast from one NPU to other NPUs.

* Model Forward and Backward Time: Specifically refers to the forward and backward of the deep learning model, which contains the forward data computation and the reverse data differential derivation.

* Optimizer time: usually refers to the model parameter update time.

* Model post-processing time: generally refers to the time after the optimizer is updated, including post-processing of data or some necessary synchronization operations, usually depending on model-specific operations.

* Communication time: a broad concept, we generally categorize the communication time as the inter-card communication elapsed time for single nodes and the inter-node communication elapsed time for multiple nodes. With the parallelization technique included in MindSpore, communication and computation can usually be executed in parallel, at which time part of the communication time is masked, so we generally consider the communication time that is not masked by computation.

* Scheduling time: This refers to the time it takes for the model to go from an instruction of the CPU to invoking a core on the NPU side.

Performance tuning that is, through the optimization of model algorithms, parameters, optimization of parallelism strategy and other means to reduce the time of the above parts, generally focusing on the optimization of the model forward-backward time, communication time.

## Introduction to Performance Tuning Basics

### Performance Indicators

Performance is usually evaluated by metrics such as throughput, arithmetic utilization (MFU and HFU).

#### Throughput

For the large language model, the throughput mainly looks at the number of tokens consumed per card per second. The formula is as follows:

$$
Throughput = SeqLength * (sample/s/p)
$$

The result of the calculation of (sample/s/p) can be obtained directly from the log, or the corresponding fields can be obtained separately from the log and then calculated.

The meaning of each field is as follows:

* SeqLength: refers to the length of the sequence, for text processing, we need to convert the input text into a sequence of numbers, and then use these number sequences as input to the model. SeqLength is the length of these number sequences, which is the length of the text. During model training and inference, we need to specify a fixed SeqLength for batch processing and computation. A longer SeqLength improves the accuracy of the model, but increases computation and memory consumption, while a shorter SeqLength reduces computation and memory consumption, but may decrease the accuracy of the model.

* sample: its value is equal to global_batch_size. in distributed training, the data is divided into multiple parts, and each part is sent to a different NPU for computation. The batch size on these NPUs adds up to the global batch size. The choice of global batch size is an important decision because it directly affects the training performance of the model. If the global batch size is too small, the batch size on each NPU may be too small, resulting in slower convergence of the model. If the global batch size is too large, the batch size on each NPU may be too large, resulting in either a lack of NPU memory or a decrease in the accuracy of the model. A good rule to find the optimal Batch Size is to reach the NPU's memory limit for a given data type, i.e., the Batch Size fills up the NPU memory.

* s: i.e., per_step_time, refers to the time spent on each step in the training process.

* p: i.e., parallel_num, data parallel dimension size.

#### Computing Capability Utilization

MFU is used to measure the current computing capability utilization without considering the case of recalculation, the higher MFU indicates the better the current computational efficiency, and the statistics are mainly the computation of GEMM. The calculation formula is as follows:

$$
MFU = \frac{FLOPs}{StepTime * HardwareCapacity}
$$

HFU takes into account the recomputation calculations in backpropagation based on MFU:

$$
MFU = \frac{FLOPs_{recompute}}{StepTime * HardwareCapacity}
$$

FLOPs (floating point operations) represent the number of floating point operations and measure the size of the computation. For example, to calculate matrix A(m,n)\*B(n,p), m\*n\*p multiplication operations and m\*n\*p addition operations are required, totaling 2\*m\*n\*p floating point operations, i.e., FLOPs are 2mnp.

StepTime refers to the time spent on each step in the training process, and HardwareCapacity is the nominal computing capability of the chips in the cluster.

When counting the computation of the transformer layer, according to the chain rule, in the backpropagation, MatMul needs to derive $d_x$ and $d_w$ separately, so the computation of the backward process is about twice as much as that of the forward one. We only need to calculate the computational amount of the forward propagation process and then estimate the computational amount of the backward propagation.

Here the GPT structure is used as an example for the theoretical estimation:

|                          | Memory Usage Byte                                      |
| ------------------------ | ------------------------------------------------- |
| No recomputation model flops      | 72 *  bLs$h^2$ * [1*corr +  s/(6h) + v/(12hL)]    |
| Selective recomputation hardware flops | 72 *  bLs$h^2$ * [1*corr + 4/3s/(6h) + v/(12hL)]  |
| Full recomputation hardware flops | 72 *  bLs$h^2$ * [4/3*corr+ 4/3s/(6h) + v/(12hL)] |

where corr = (60+12/q)/72, q is the multiplicity of GQA, and q = n_heads/n_kv_heads. The reason for the 2-fold increase from recomputation is that the forward direction of Q, K, and V in Attention needs to be recomputed. If only the forward direction needs to be recomputed, it should become 4b$s^2$h(forward) + 8b$s^2$h(reverse) = 12b$s^2$h based on the non-recomputed 4b$s^2$h(forward) + 8b$s^2$h(reverse) + 4b$s^2$h(forward) = 16b$s^2$h. The recomputation increases the overhead by 16/12 = 4/3 times.

The detailed calculation steps are:

| Module                      | Specifications                               | FLOPS                                  |
|-------------------------|----------------------------------|----------------------------------------|
| attention               |                                  |                                        |
| Query, key, Value  MatMul | [b, s, h] * [h, h]               | (2+4/q)*bs$h^2$   q=n_heads/n_kv_heads |
| QK BatchMatMul          | [b, a, s, h/a] *  [b, a, h/a, s] | 2b$s^2$h                               |
| score \* V              | [b, a, s, s] * [b, a, s, h/a]    | 2b$s^2$h                                  |
| attention projection    | [b, s, h] * [h,h]                | 2bs$h^2$                               |
| MLP                     |                                  |                                        |
| MLP mapping             | [b, s, h] * [h, 4h]              | 8bs$h^2$                               |
| MLP projection          | [b, s, 4h] * [4h, h]             | 8bs$h^2$                               |
| LmHead                  |                                  |                                        |
| lmHead projection       | [b, s, h] * [v, h]               | 2bshv                                  |
| Total                   |                                  | 2bshv                                  |
| GPTlayer-Total          |                                  | (20+4/q)bs$h^2$ + 4b$s^2$h+2bshv          |

The meaning of each character is as follows:

* b：micro batch size
* h：hidden size
* s：seq length
* v：vocab size
* L：layers

The Llama structure (gated FFN, 8-way GQA) is slightly different from the GPT, and differs from the GPT mainly in that there are differences in the mlp layer, and in the Gated MLP used in the Llama series, the specific flops are calculated as follows:

| MLP mapping      | [b, s, h] * [h, $\hat{h}$]           | 2bsh$\hat{h}$                                |
| ---------------- | ------------------------------------ |----------------------------------------------|
| MLP gate         | [b, s, h] * [h, $\hat{h}$]           | 2bsh$\hat{h}$                                |
| MLP projection   | [b, s, $\hat{h}$]  * [$\hat{h}$,  h] | 2bsh$\hat{h}$                                |
| Total            |                                      | 6bsh$\hat{h}$                                |
| Llamalayer-Total |                                      | (4+4/q)bs$h^2$ + 4b$s^2$h+6bsh$\hat{h}$ + 2bshv |

Note: $\hat{h}$ is ffn hidden size

The complete MFU estimates for the Llama series are given below:

![Llama_memory](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/llama_memory.png)

HFU/MFU can be used for the evaluation of tokens/s/p for training performance. Generally HFU>50% is considered as better hardware utilization, for example, Llama2-7B is 4695tokens/s/p, which corresponds to MFU=57%, HFU=65% and is considered as more desirable results. For large-parameter models such as Llama2-70B, the MFU/HFU decays at a linear ratio as the parallel scale expands. [PaLM](https://arxiv.org/pdf/2204.02311.pdf) counts the MFU for several common large models.

![PaLM-MFU](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/PaLM-MFU.png)

### Introduction to Parallel Feature

In large model training, due to the increase of data volume and model complexity, the computational capacity of a single computing node is difficult to meet the training demand. In order to improve the training efficiency and accelerate the training process, a parallel strategy is usually used to distribute the computational tasks to multiple computational nodes for computation.

Parallelism strategies are usually classified into various parallel modes such as Data Parallelism (DP), Model Parallelism (generally referred to as Tensor Parallelism (TP)), Pipeline Parallelism (PP), Optimizer Paralleism (OP), Sequence Parallelism (SP), and Multi-Copy Parallelism. In practice, multiple parallel strategies, as well as multiple optimizations, such as using optimizer parallelism and recomputation, are usually employed to reduce the model's use of memory and improve training efficiency. Parallel strategy design is closely related to the efficiency of the model, and it is crucial to identify one or more sets of better parallel strategies before model tuning.

For details, refer to [Parallel Strategy Guide](https://www.mindspore.cn/mindformers/docs/en/dev/function/distributed_parallel.html).

### Memory Analysis

The dominant structures of the large models are transformer decoder only structures, which consist of two sublayers, self attention and ffn. A typical model Llama2 is shown below:

![llama_layer](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/llama_layer.png)

#### Static Memory

Static memory is typically used to store model parameters, and the number of model parameters is the number of parameters that can be learned and tuned in a neural network or machine learning model. These parameters include weights and biases, which are continuously updated during training to optimize the model's performance.

Taking the GPT structure as an example, the number of parameters of a transformer layer is shown below:

![static_memory](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/static_memory.png)

The static memory mainly contains the parameters of the model and the state of the optimizer, if the gradient accumulation or pipeline parallelism is enabled, there will be one more gradient; setting N as the number of model parameters and t as the size of the optimizer parallelism (the default is equal to DP). The memory occupancy for each scenario is as follows:

|                     | Static Memory Usage       | Descriptions                                                          |
| ------------------- | ------------------ |-------------------------------------------------------------|
| no parallel              | 2N + 4N + 4N = 10N | FP16 weight (2N)  FP32 AdamVar (4N)  FP32 AdamMomentum (4N) |
| Optimizer Parallel          | 10N / t            | 权重和优化器状态切分                                                  |
| Optimizer Parallel+gradient accumulation | 10N / t + 2N       | Weight slicing and optimizer state slicing, gradient accumulation not slicing.                                      |

For example, a 7B model, if t = 8, the theoretical predictions of static memory for the above three scenarios are 70GB, 8.75GB, and 22.75GB, respectively.

#### Dynamic Memory

Dynamic memory is typically used to store the results of intermediate computations.

Taking GPT as an example and considering PP, TP, etc., the theoretically calculated values are as follows:

|                              | Memory[Byte]              | Descriptions                                              |
| ---------------------------- | ----------------------- | ------------------------------------------------- |
| no parallel                       | sbh * [34+5as/h]        | total memory                                            |
| Model Parallel                     | sbh * [10+(24+5as/h)/t] | attn/ffn inputs, 2*ln inputs, 2dropout masks are not being parallelized. |
| Model Parallel+Sequence Parallel            | sbh * [34+5as/h]/t      | Further memory reduction in dimension t                             |
| Model Parallel+selective recomputation          | sbh * [10+24/t]         | recompute $s^2$ memory                               |
| Model Parallel+Sequence Parallel+selective recomputation | sbh * [34/t]            | recompute $s^2$ memory                               |
| full recomputation               | sbh * [2]               | Save only the inputs for each layer                                  |

The detailed calculation steps are:

| Module                       | Variables that need to be saved reversely      | Memory size[Byte] |
|--------------------------| --------------------- | -------------- |
| Attention part           |                       |                |
| Query, key, Value MatMul | x                     | 2sbh           |
| QK BatchedMatMul         | Q, K                  | 4sbh           |
| softmax                  | softmax  result       | 2a$s^2$b          |
| softmax dropout          | dropout  mask         | a$s^2$b           |
| prob-value BatchedMatMul | dropout  result and V | 2a$s^2$b + 2sbh   |
| attenton projection      | dropout  mask+output  | sbh + 2sbh     |
| Totoal                   |                       | 11sbh + 5a$s^2$   |
| FFN part                    |                       |                |
| MLP  mapping             | x                     | 2sbh           |
| MLP  activation          | hidden                | 8sbh           |
| MLP  projection          | activated  hidden     | 8sbh           |
| MLP  dropout             | dropout  mask         | sbh            |
| Totoal                   |                       | 19sbh          |
| LayerNorm part             |                       |                |
| two lanyernorm           | input                 | 2sbh + 2sbh    |

The meaning of each character is as follows:

* a：number of attention heads
* b：micro batch size
* h：hidden size
* L：number of transformer layers
* p：pipeline parallel size
* s：seq length
* t：tensor parallel size
* v：vocab size

#### Mixed Precision

Floating point data types are mainly categorized into double precision (FP64), single precision (FP32), and half precision (FP16). In the training process of the neural network model, the single precision (FP32) floating point data type is generally used by default to represent the network model weights and other parameters.

FP16 has half the storage space of FP32, and similarly, FP32 is half the size of FP64. Therefore, the use of FP16 for computing has the advantages of less memory occupation, higher computational efficiency and communication efficiency, but the use of FP16 also brings problems such as data overflow and rounding errors.

While using mixed precision to gain training speedup and memory savings, the solution to the problem introduced by FP16 needs to be considered. The main idea of Loss Scale, a solution to the FP16 type data overflow problem, is to expand the loss by a certain number of times when calculating the value loss. According to the chain rule, the gradient will be expanded accordingly, and then scaled down by the corresponding multiple when the optimizer updates the weights, thus avoiding data underflow.

For details, refer to [automatic mixed precision](https://www.mindspore.cn/tutorials/en/master/beginner/mixed_precision.html)。

### Tools Introduction

#### profiler Tool

MindFormers itself integrates profiling data collection with the following steps:

1. Modify the configuration files

   Turn on the profiling switch in the model configuration file (e.g. run_llama2_7b.yaml) with the following parameters to be changed:

   ```yaml
   profile: True  # Whether to enable performance analysis tools
   profile_start_step: 1  # Step that starts performance analysis
   profile_stop_step: 10  # Step that ends performance analysis
   init_start_profile: False  # Enabled when Profiler is initialized, profile_start_step will not take effect after it is enabled.
   profile_communication: True # Whether to collect communication performance data in multi-NPU training
   profile_memory: True  # Collect Tensor memory data
   ```

2. Simplify model

   It is recommended that the number of layers (num_layers) of the model be changed to 2 to facilitate fast data collection.

3. View Data

   The collection tool will create a profile folder under the model configuration file output_dir (default “. /output") to generate performance data for each card of the current machine according to the rank id. Taking rank_0 for example, the directory is output/profile/rank_0/profiler.

   The generated file and its introduction refer to [Introduction to profile file](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html#directory-structure), which mainly collects information such as running time of operators and tasks, CPU utilization and memory consumption, and all data required in performance tuning analysis.

#### MindStudio Insight

MindStudio Insight provides multiple presentations of performance data, including visual presentations of Timeline views, communication analysis, computational elapsed time, so that users can analyze potential performance bottlenecks and provide guidance on how to take steps to eliminate or reduce them. MindStudio Insight supports viewing data exported by Profiling in Timeline for cluster scenarios and displaying it in a single-card dimension, and can support cluster performance file analysis of more than 20GB.

Click [MindStudio Insight download link](https://www.hiascend.com/zh/developer/download/commercial/result?module=sto) and select the appropriate version to install.

Open MindStudio Insight, click the “plus sign” in the toolbar at the top left of the interface, select the file or directory to be parsed and exported in the pop-up window, and then click “Confirm” to import.

MindStudio Insight tool provides users with the full process of online inference, training process operation in the timeline (Timeline) presentation, and in accordance with the scheduling process to present the overall operating conditions, and MindStudio Insight support cluster Timeline display. By analyzing the timeline, users can analyze the online inference/training process at a fine-grained level, such as whether the iteration gap is too long, the execution time of operators, and provide some easy-to-use functions to assist users to quickly locate the performance bottlenecks.

The Timeline interface consists of four parts: the toolbar (Area I), the timeline tree (Area II), the graphical pane (Area III), and the data pane (Area IV), as shown in the figure.

![studio](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/studio.png)

* Area I

  The toolbar, which contains frequently used shortcut buttons, from left to right, is a list of markers, filtering (supports filtering displays by card or by special layer), operator search, operator linking, page adaption and zoom buttons.

* Area II

  Timeline tree diagram showing the hierarchical information of each “Card” in the cluster scenario, with “Card” at the first level, process or specialization hierarchies at the second level, and threads and other names at the third level. This includes upper application data (containing elapsed time information of upper application arithmetic), CANN layer data (containing elapsed time data of AscendCL, GE, and Runtime components), underlying NPU data (containing elapsed time data and iteration trajectory data of each Stream task flow under Ascend Hardware, HCCL and Overlap Analysis communication data, and other Rise AI processor system data), hitpoint data, and the AI Core Freq hierarchy.

* Area III

  The graphical pane, which displays data within an iteration, corresponds to a timeline tree diagram, which provides a row-by-row graphical presentation of the timeline, including the execution sequence and execution duration of the upper-level application operators, components and interfaces.

* Area IV

  Data pane, statistical information or operator detail information display area, Slice Detail for detailed information on selected individual operators, Slice List for a list of operators in the selected area of a lane, and System View for a summary of operators in a category.

Click anywhere on the timeline page tree or graphical pane can be performed using the W (zoom in), A (move left), S (zoom out), and D (move right) keys in the keyboard, which support zooming in with a maximum precision of 1ns. This tool can provide overview, memory, arithmetic, communication and other dimensions of analysis to assist in performance tuning. Refer to [MindStudio Insight User Guide](https://www.hiascend.com/document/detail/zh/mindstudio/70RC2/msinsightug/msascendinsightug/AscendInsight_0002.html) for detailed usage.

#### IR Graph

In the MindFormers configuration file, just turn on save_graphs, and the runtime will output some intermediate files ending with the ir suffix generated during the graph compilation process, which we call IR files. By default, a directory of graphs will be generated in the current task execution directory, and all IR graphs will be saved in this. It is a relatively intuitive and easy to understand document describing the structure of the model in text format, which can be viewed directly with text editing software. Refer to [Config Configuration Description](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html) for the meaning of the configuration items, and the configuration method is as follows:

```yaml
context:
  save_graphs: True
  save_graphs_path: "./graph"
```

An excerpt of some of the IR graph:

```text
  %13(equiv_180_CNode_16165) = Load(%para6_model.layers.0.attention.wq.weight, UMonad[U]) cnode_attrs: {checkpoint: Bool(1)} cnode_primal_attrs: {unique_id: "782039"}
      : (<Ref[Tensor[Float16]], (512, 4096), ref_key=model.layers.0.attention.wq.weight>, <UMonad, NoShape>) -> (<Tensor[Float16], (512, 4096)>)
      # Fullname with scope: (Default/network-MFPipelineWithLossScaleCell/network-_VirtualDatasetCell/_backbone-GradAccumulationCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LLamaDecodeLayer/attention-LLamaAttention/Load-op0)
  %14(equiv_16877_x) = PrimFunc_MatMul(%12, %13, Bool(0), Bool(1)) {instance name: matmul} primitive_attrs: {in_strategy: ((1, 1), (8, 1))} cnode_attrs: {checkpoint: Bool(1)} cnode_primal_attrs: {unique_id: "782146", origin_output_shape: (4096, 4096), micro: I64(0), origin_input_shapes: ((4096, 4096), (4096, 4096))} {in_strategy: ((1, 1), (8, 1))}
      : (<Tensor[Float16], (4096, 4096)>, <Tensor[Float16], (512, 4096)>, <Bool, NoShape>, <Bool, NoShape>) -> (<Tensor[Float16], (4096, 512)>)
      # Fullname with scope: (Default/network-MFPipelineWithLossScaleCell/network-_VirtualDatasetCell/_backbone-GradAccumulationCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LLamaDecodeLayer/attention-LLamaAttention/wq-Linear/MatMul-op0)
  %15(equiv_16876_CNode_30913) = PrimFunc_Reshape(%14, (I64(1), I64(4096), I64(4), I64(128))) {instance name: reshape} cnode_attrs: {checkpoint: Bool(1)} cnode_primal_attrs: {unique_id: "817859", forward_comm_node_unique_id: "729440", micro: I64(0)}
      : (<Tensor[Float16], (4096, 512)>, <Tuple[Int64*4], TupleShape(NoShape, NoShape, NoShape, NoShape), elements_use_flags={[const vector]{1, 1, 1, 1}}>) -> (<Tensor[Float16], (1, 4096, 4, 128)>)
      # Fullname with scope: (Default/network-MFPipelineWithLossScaleCell/network-_VirtualDatasetCell/_backbone-GradAccumulationCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LLamaDecodeLayer/attention-LLamaAttention/Reshape-op0)
  %16(equiv_16875_query) = PrimFunc_Transpose(%15, (I64(0), I64(2), I64(1), I64(3))) {instance name: transpose} primitive_attrs: {in_strategy: ((1, 1, 8, 1))} cnode_attrs: {checkpoint: Bool(1)} cnode_primal_attrs: {unique_id: "782042", micro: I64(0)} {in_strategy: ((1, 1, 8, 1))}
      : (<Tensor[Float16], (1, 4096, 4, 128)>, <Tuple[Int64*4], TupleShape(NoShape, NoShape, NoShape, NoShape), elements_use_flags={[const vector]{1, 1, 1, 1}}>) -> (<Tensor[Float16], (1, 4, 4096, 128)>)
      # Fullname with scope: (Default/network-MFPipelineWithLossScaleCell/network-_VirtualDatasetCell/_backbone-GradAccumulationCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LLamaDecodeLayer/attention-LLamaAttention/Transpose-op0)
```

`%XX` indicates the step, followed by the name of the operator, and the parentheses contain the inputs and outputs, while Fullname with scope contains the completed class, method name, and so on.

* `%13`

  This step loads wq.weight directly and gets <Tensor[Float16], (512, 4096)>.

* `%14`

  MatMul with the previous %12 output and the %13 output above to get <Tensor[Float16], (4096, 512)>.

* `%15`

  For the 14% output above, Reshape gets <Tensor[Float16], (1, 4096, 4, 128)>.

* `%16`

  For the 15% output above, Transpose gets <Tensor[Float16], (1, 4, 4096, 128)>.

It is recommended to change the number of layers of the model to a smaller size when saving IR graph, to reduce the time of compiling and saving graph, and to facilitate fast debugging. For details, please refer to [Introduction to IR file](https://www.mindspore.cn/docs/en/master/model_train/debug/error_analysis/mindir.html#ir-introduction) and [Analysis samples](https://www.mindspore.cn/docs/en/master/model_train/debug/error_analysis/mindir.html#how-to-derive-the-cause-of-the-failure-based-on-the-analyze-fail-ir-file-analysis-graph).

## Performance Tuning Guide

### Overall Concept

The performance optimization of large models mainly relies on profiling data analysis and memory analysis to analyze the current performance bottlenecks and the gap with competitors. The time consumption on the MindSpore framework mainly consists of operator time consumption and communication time consumption, in which the operator time consumption is mainly to dismantle the core operators and the gap with competitors, and the communication analysis is to see whether there is an irrational rearranged distribution, and so on.

After completing the performance data as well as memory data collection, the overall optimization flow is as follows:

* Analyze the performance of operators and try to use fusion operators to replace small and medium-sized operators.

* Analyze the communication time consumption, check whether there exists a better distributed strategy, analyze whether there exists an unreasonable rearranging problem, and improve the efficiency of the whole cluster.

* Analyze the memory, check whether there is an abnormally large memory Tensor, whether there is a fusible operator to reduce the memory. In the case of memory affluence you can explore the choice of recomputation settings, or reduce the number of copies of the model slicing to reduce the communication overhead caused by the model slicing.

Performance optimization is a cyclic process, as shown in the figure below, after the operator optimization is completed, it is necessary to conduct experimental analysis of the cluster distributed strategy, to analyze whether the communication time consumption is reasonable, whether there is additional re-arrangement of the distribution of the overhead; and then carry out the analysis of the memory optimization, after the memory optimization is completed, whether it can be re-adjusted to the cluster strategy settings, so as to obtain a more optimal set of policies. The cycle is repeated to optimize, and then step by step to achieve the set performance goals.

![process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/process.png)

After completing a round of performance optimization, it is also necessary to ensure that the model accuracy is aligned, and the alignment applies this optimization strategy.

### Operator Performance Optimization

#### Comparison of GPU and NPU data

* Basic idea

  NPU and GPU operator elapsed time statistics, get data by profiling.

  Compare GPU and NPU data to find the difference of single operator time consumption.

  Detailed performance analysis, analyze the performance of a single case according to the operator PMU data, and accurately identify the operators to be optimized.

* Operator time consumption statistics

  NPU operator elapsed time statistics can be obtained directly from profiling, which analyzes the current major time consumption operators as well as the inefficient ones, so as to find out the operators that need to be optimized. Refer to [Introduction to using profiler tool](#profiler-tool).

  GPU operator elapsed time statistics, refer to [PyTorch Training Profiling Analysis Method](https://www.hiascend.com/document/detail/zh/mindstudio/60RC1/quickstart/systemprofilerpt_000001.html) provided by MindStudio.

* Data dismantling concerns

  In the performance data generated by the above [profiler tool](#profiler-tool), analyze the following file in the generated data: rank-*_ascend_ms/ASCEND_PROFILER_OUTPUT/kernel_details.csv. This file counts the overall percentage of a certain type of operator by operator type, from which we can get which type of operator needs to be optimized to bring about a performance improvement.

Operator fusion is used to reduce runtime memory access and improve computational efficiency by combining multiple independent operators into a larger and more complex one. It can reduce the storage and transmission of intermediate results, effectively reducing the memory overhead. In addition, combining multiple operators can reduce the number of computations, which can effectively improve the computational efficiency on the NPU.

Currently MindFormers automatically performs fusion operator optimization by default, automatically merging multiple consecutive small operators in the model that meet the conditions into a single fusion operator.

### Communication Optimization

In the semi-automatic parallel development mode, the developer is required to configure the parallel slicing strategy for the input Tensor and output Tensor of each operator. If there is a mismatch of operator configurations, it will result in the MindSpore framework inserting communication operators at compile time and rearranging the Tensor to fit the subsequent operator's slicing method. Common communication operators are AllGather, AllReduce, and so on.

The timeline.json captured by Profiling is analyzed by a visualization tool, and then contextual analysis is performed based on the operator number in conjunction with the IR graph to check whether the communication operator here matches the configured cut strategy.

Use [profiler tool](#profiler-tool) to generate the file ascend_timeline_display_0.json, and then open the file by typing “chrome://tracing” in Chrome, or you can use [MindStudio Insight](#mindstudio-insight) to open it and parse out the corresponding timing diagram of the computational communication task flow. This is shown below:

![timeline](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/timeline.png)

A computational gap exists after the wo-linear, and in conjunction with the IR diagram,

It can be seen that there exists an AllReduce communication operator after the wo-linear and AllReduce receives the output of the MatMul of the wo-linear. The IR diagram is shown below:

```text
%100(equiv_loss)) = MatMul(%98, %99) {instance name: matmul) primitive_attrs: {IsFeatureMapInputList: (0), input names: [x1, x2], transpose_x2: Bool(1), transpose_b: Bool(1), in strategy: ((1, 4), (1,4)), output names: [output], transpose_a: Bool(0), IsFeatureMapOutput: Bool(1), transpose_x1: Bool(0)} cnode_attrs: (checkpoint: Bool(1), is dynamic_len: Bool(0)} cnode_primal_attrs: (unique id: "230416", micro: I64(0))}
    : (<Tensor[Float16], (2048, 1024)>, <Tensor[Float16], (4096, 1024)>) -> (<Tensor[Float16], (2048, 4096)>)
    # Fullname with scope:
    (Default/network-_VirtualDatasetCell/_backbone-GradAccumulationCell/network-LlamaForCausalLM/model-LlamaModel/lavers-Celllist/1-LLamaDecodeLaver/attention-LhamaAttention/wo-Linear/MatMul-op8)
%101(equiv_CNode_3914) = AllReduce(%100) {instance name: forward_op_6937918150211178578} primitive_attrs: {IaFeatureMapInputList: (0), comm_reuse: Bool(1), group: "hcel_world_group", fusion: 164(0), op: "sum", rank_list: (0, 1, 2, 3), group_ranks: "WORLD_GROUP", index: 164(0), group_rank ids: (0, 1, 2, 3), IsFeatureMapOutput: Bool(1), _parallel_group: "hcel_world_group", no_eliminate: Bool(1)} cnode_attrs: {checkpoint: Bool(1), is_dynamic_len: Bool(0)} cnode_primal_attrs: {unique_id: "231701", forward_comm_node_unique_id: "224263", micro: I64(0)}
    : (<Tensor[Float16], (2048, 4096)>) -> (<Tensor[Float16], (2048, 4096)>)
    # Fullname with scope:
    (Default/network- VirtualDatasetCell/ backbone-GradAccumulationCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/1-LLamaDecodeLayer/attention-LLamaAttention/wo-Linear/AllReduce-op0)
```

It can be found that MatMul operator for both inputs are sliced, the first input is sliced in columns, and the second input is sliced in rows, which is in line with the TensorParallel Row Parallel Linear slicing. At this time if we want to maintain the MatMul mathematical equivalence needs on the results of the computation of the AllReduce operation. At this point, the AllReduce inserted at this time is in line with expectations.

### Parallel Strategy

#### Features of Parallel Strategy

The features of different parallel strategies are summarized below:

* Data parallelism

  Multiple pieces of data are trained at the same time and communicated only once at the gradient update for optimal performance without memory reduction;

* Model Parallelism

  Slicing the whole model into different Devices, the network computes the respective parts in parallel and communicates at locations such as LayerNorm, which saves the most memory but has a large amount of communication;

* Pipeline Parallelism

  Slices different stages of the model into different Devices, the network computes the respective stages serially and communicates when switching stages, saves some memory by recomputing, less communication, but there will be computational idleness (bubble);

* Optimizer parallelism

  Slicing the optimizer weights, model weights by DP (DP can exactly divide the 0th dimension of the weights shape) and communicating when the gradient is updated, which can save memory significantly and the communication is small;

* Sequence parallelism

  Short sequence parallelism slices the sequence dimension by MP at LayerNorm, unchanged communication, reducing memory and some computation of Norm;

* Multi-copy parallelism

  In model parallelism, MatMul and other operators are sliced into multiple copies, and the computational communication between different copies is interleaved to achieve communication masking;

#### Suggestions

In practice, multiple parallelization strategies are usually used in combination. According to the model size, the number of machines to determine the appropriate parallelism strategy. This section describes the recommended configurations for different model sizes. The meanings of the configuration items in the sample configurations are referenced in [Config Configuration Description](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html)。

* Small parameter models

  When the model size is small (e.g. 7B), pure data parallelism + optimizer parallelism can be used, and gradient accumulation can be further turned on if memory is rich. Use 8-card training, [Llama2-7B parallelism strategy recommended configuration](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_7b.yaml).

* Moderate parameter models

  For moderate model sizes (e.g. 13B), further pipeline parallelism can be used and recomputation can be tuned. Use 8-card training, [Llama2-13B parallel strategy recommended configuration](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_13b.yaml).

* Large parameter models

  When the model size is large (e.g., 70B), model parallelism needs to be turned on, while sequence parallelism and multicopy parallelism are also recommended. Use 64-card training, [Llama2-70B parallel strategy recommended configuration](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_70b.yaml).

### Recomputation

#### cast Recomputation

RmsNorm is generally computed using float32, and the input needs to be Cast from fp16 or bf16 to fp32 before computation; RmsNorm needs to save the input for reverse computation. Therefore, recalculating Cast from fp16 to fp32 can save memory by changing the memory from the input of RmsNorm to the input of Cast, which is half the size of the input of RmsNorm.

![cast](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/cast.png)

Doing recalculation from high precision to low precision Cast operators will result in the later operators originally only needing to store the low precision memory after cast, and after the Cast operators are recalculated, they need to store the high precision memory, which will instead result in a larger memory.

#### Silu-Mul Recomputation

In FeedForward, the middle part of the memory tends to be large. Silu and Mul recomputation is less costly. Recomputing the Silu and Mul operators saves memory for the first inputs of MatMul and Mul of w2.

![silu_mul](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/silu_mul.png)

#### Communication Recomputation

With sequence parallel is turned on, RmsNorm slices at the sequence dimension, and then aggregates Tensor from different cards via AllGather for later MatMul computation. If AllGather is recalculated, each card only needs to store one copy of the memory before AllGather to achieve the effect of memory reduction.

![communicate](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/communicate.png)

### Memory Optimization

#### NPU Memory Peak Analysis

Peak memory is an important concern when analyzing memory. When executing training, set the following environment variables to count the memory basics.

```shell
export MS_MEMORY_STATISTIC=1
```

When training is complete, the following message is output at the end of the log file:

```text
Device HBM memory size: 62432M
MindSpore Used memory size: 59392M
MindSpore memory base address: 0
Used peak memory usage (without fragments): 48874M
Actual peak memory usage (with fragments): 48874M
```

Used peak memory usage (without fragments) indicates the peak NPU memory usage without fragments.

Actual peak memory usage (with fragments) represents the peak NPU memory usage with fragments.

Before formal training, you can use dryrun to simulate the training, which requires only one card to simulate the overall NPU memory peak. dryrun script is as follows:

```shell
export ENABLE_CELL_REUSE=1
export MS_MEMORY_STATISTIC=1
export MS_SIMULATION_LEVEL=1
export RANK_SIZE=16
export RANK_ID=0
python run_mindformer.py --config ${CONFIG} --run_mode train > dry_run.log 2>&1 &
```

RANK_SIZE indicates the number of cards to be simulated, the RANK_ID indicates the card to be simulated, and the three environment variables, ENABLE_CELL_REUSE, MS_MEMORY_STATISTIC, and MS_SIMULATION_LEVEL, are used to set the dryrun mode.

### Typical Case

#### Silu-Mul Recomputation Not in Effect

Performing recomputation on Silu and Mul saves memory when fine-grained multicopy is on, but doing recomputation on Silu and Mul does not save memory when fine-grained multicopy is off. The localization process is as follows:

* Confirmation that recomputation is configured

  Check if the Cast, Silu and Mul operators have the label "recompute: Bool(1)" in the IR graph. If they do, it means that the operators are equipped with recompute.

* Checking for recomputation operators

  The IR graph is checked for operators with duplicated labels for Cast, Silu, and Mul. The absence of labeled operators indicates that the actual computational graph does not recompute this part of the operator. Here only Cast operator is with duplicated label.

  ```text
  %1834(CNode_108839) = PrimFunc_Cast(%1833, I64(43)) {instance name: cast} primitive_attrs: {output_names: [output], input_names: [x, dst_type], recompute: Bool(1)} cnode_attrs: {recompute_sub_graph: U64(64), recompute_id: I64(65), duplicated: Bool(1), need_cse_after_recompute: Bool(1)} cnode_primal_attrs: {micro: I64(0)}
      : (<Tensor[Float16], (1, 4096, 4096)>, <Int64, NoShape>) -> (<Tensor[Float32], (1, 4096, 4096)>)
  ```

* Checking the reverse calculation input

  The inputs to the reverse operators of Silu and Mul are checked in the IR diagram to see if they are as expected, and there are Reshape operators between Silu and Mul, and between Mul and MatMul when fine-grained multicopy is off, and Silu, Mul, and MatMul are connected when fine-grained multicopy is on. The process is as follows:

![reshape](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindformers/docs/source_zh_cn/perf_optimize/images/reshape.png)

It can be seen that the cause is that the input shape of Linear in the fine-grained multicopy scenario is two-dimensional, while the input shape of Linear in the non-fine-grained multicopy scenario is three-dimensional, resulting in a Reshape operator between Linear and Mul, and the lack of Reshape recalculation results in recalculation of Silu alone being optimized. The additional recalculation of the Reshape results in a normal memory reduction. The reference configuration is as follows:

```yaml
recompute_config:
  recompute: False
  select_recompute: ['feed_forward\.mul', 'feed_forward\.w1\.activation', 'feed_forward\.w1\.reshape', 'feed_forward\.w2\.reshape']
```

#### Llama2-13B Extreme Performance Optimization

13B defaults to a single DP: 8, MP: 1, PP: 1 with full recalculation on, with performance around 1860tokens/s/p and 40% MFU, which is significantly lower compared to the 7B (53% MFU) & 70B (47% MFU).

After analyzing, 13B performance bottleneck mainly lies in memory, whether single or multi-computer, if you don't slice MP, you need to turn on full recalculation, and doing selective recalculation for Silu and Mul memory is still not enough; full recalculation will be an additional 20% to 25% more computation, resulting in low performance; MP slices can be turned off the recalculation, but the performance is a little lower than the pure DP.

Adjusting the sharding strategy to DP: 8, MP: 1, PP: 2, micro: 128 with dual machines and full recomputation on improves performance to 2136tokens/s/p. Changing the full recomputation to selective recomputation and fine selecting the operators to minimize the amount of memory at each layer improves performance to 2189tokens/s/p.

```yaml
select_recompute: ['feed_forward\.mul', 'feed_forward\.w1\.activation', 'feed_forward\.w1\.reshape', 'feed_forward\.w1\.matmul', 'feed_forward\.w3\.matmul', 'feed_forward\.W3\.reshape', 'feed_forward\.w2\.matmul', 'feed_forward\.w2\.reshape', 'ffn_norm\.norm', 'ffn_norm\.rcast', 'attention_norm\.norm', 'attention_norm\.rcast', 'attention\.wq\.reshape', 'attention\.wk\.reshape', 'attention\.wv\.reshape', 'attention\.wo\.matmul', 'attention\.wo\.reshape', 'attention\.merger_head_transpose', 'add', 'attention\.flash attention']
```

Adjusting the number of recomputation layers for different stages results in less recomputation for stage1 and performance improvement to 2210tokens/s/p.

```yaml
select_recompute:
  'feed_forward\.mul': [20, 8]
  'feed_forward\.w1\.activation': [20, 8]
  'feed_forward\.w1\.matmul': [20, 0]
  'feed_forward\.w1\.reshape': [20, 8]
  'feed_forward\.w3\.matmul': [20, 0]
  'feed_forward\.w3\.reshape': [20, 0]
  'feed_forward\.w2\.matmul': [20, 0]
  'feed_forward\.w2\.reshape': [20, 0]
  'ffn_norm\.norm': [20, 0]
  'ffn_norm\.rcast': [20, 0]
  'attention_norm\.norm': [20, 0]
  'attention_normi.rcast': [20, 0]
  'attention\.wq\.reshape': [20, 0]e
  'attention\.wk\.reshape': [20, 0]e
  'attention\.w\.reshape': [20, 0]e
  'attention\.wol.matmul': [20, 0]
  'attention\.wo\.reshape': [20, 0]e
  'attention\.merger head transpose': [20, 0]
  'add': [20, 0]
  'attention\.flash_attention': [20, 0]
```

Using graph compilation level of O0/O1 graph kernel fusion, there are further optimizations in memory, changing the selective recomputation of most of the operators to full recomputation of some layers, and configuring selective recomputation of Silu and Mul for the rest of the layers. The number of fully-recomputed layers in stage0 and stage1 is 13 and 5 respectively, and the performance improves to 2,353tokens/s/p. Gradually the number of fully-recomputed layers in stage0 and stage1 are 4 and 0 respectively, and the performance is improved to 2562tokens/s/p (max_device_memory: 57.2GB). The reference configuration is as follows:

```yaml
recompute_config:
  recompute: [4, 0]
  select_recompute: ['feed_forward\.mul', 'feed_forward\.w1\.activation', 'feed_forward\.w1\.reshape', 'feed_forward\.w2\.reshape']
```

After the final tuning, the Llama2-13B performance was optimized to 2562tokens/s/p, MFU 55.4% and HFU 60.4%, for a total improvement of 37%.
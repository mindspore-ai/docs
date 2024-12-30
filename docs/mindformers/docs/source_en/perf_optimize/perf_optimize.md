# Large Model Performance Optimization Guide

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/perf_optimize/perf_optimize.md)

## Performance Optimization Descritions

This document introduces the performance tuning of large language models, detailing the basic theoretical knowledge related to performance tuning, guidance on the use of related tools and the overall idea of performance tuning, as well as case sharing. When you start to work on performance tuning of large models, you should have the basic knowledge of large models. In order to avoid dispersion, this document will not explain the basic concepts related to large models, and focus on performance tuning introduction.

Performance generally includes in terms of model training performance, with the time required to complete a single end-to-end training session, given a specified model and input data. End-to-end refers to the process of completing a single-step training of an AI model, and the time is mainly composed of the following components:

* Data loading time: it refers to the time for the model to load the training data and weights, including reading the data from the hardware storage device into the CPU, preprocessing the data in the CPU, and carrying the CPU data to the NPU. For some models that need to be sliced onto several NPUs, the data loading time also includes the time to broadcast from one NPU to other NPUs.

* Model Forward Computation and Backward Computation Time: contains the forward data computation and the reverse data differential derivation.

* Optimizer time: it refers to the model parameter update time.

* Model post-processing time: it refers to after the optimizer is updated, including post-processing of data or necessary synchronization operations, usually depending on model-specific operations.

* Communication time: a broad concept, including the inter-card communication elapsed time for single nodes and the inter-node communication elapsed time for multiple nodes. With the parallelization technique in MindSpore, communication and computation can usually be executed in parallel, at which time part of the communication time is masked, so we generally consider the communication time that is not masked by computation.

* Scheduling time: it refers to the time it takes for the model to go from a CPU instruction to invoking the NPU kernel.

Performance tuning that is, through the optimization of model algorithms, parameters, parallelism strategy and other means to reduce the time of the above parts, generally focusing on the optimization of the model forward-backward time, communication time.

## Introduction to Performance Tuning Basics

### Performance Indicators

Performance is usually evaluated by throughput. For the large language model, the throughput mainly looks at the number of tokens processed per card per second. The formula is as follows:

$$
Throughput = SeqLength * (sample/s/p)
$$

The result of the calculation of (sample/s/p) can be obtained directly from the log, or the corresponding fields can be obtained separately from the log and then calculated.

The meaning of each field is as follows:

* SeqLength: refers to the length of the sequence, for text processing, we need to convert the input text into a sequence of numbers, and then use these number sequences as input to the model. SeqLength is the length of these number sequences, which is the length of the text. During model training and inference, we need to specify a fixed SeqLength for batch processing and computation. A longer SeqLength improves the accuracy of the model, but increases computation and memory consumption, while a shorter SeqLength reduces computation and memory consumption, but may decrease the accuracy of the model.

* sample: its value is equal to global_batch_size. in distributed training, the data is divided into multiple parts, and each part is sent to a different NPU for computation. The batch size on these NPUs adds up to the global batch size. The choice of global batch size is an important decision because it directly affects the training performance of the model. If the global batch size is too small, the batch size on each NPU may be too small, resulting in slower convergence of the model. If the global batch size is too large, the batch size on each NPU may be too large, resulting in either a lack of NPU memory or a decrease in the accuracy of the model. A good rule to find the optimal Batch Size is to reach the NPU's memory limit for a given data type, i.e., the Batch Size fills up the NPU memory.

* s: i.e., per_step_time in seconds, refers to the time spent on each step in the training process.

* p: i.e., parallel_num, data parallel dimension size.

### Introduction to Parallel Feature

In large model training, due to the increase of data volume and model complexity, the computational capacity of a single computing node is difficult to meet the training demand. In order to improve the training efficiency and accelerate the training process, a parallel strategy is usually used to distribute the computational tasks to multiple computational nodes.

Parallelism strategies are usually classified into various parallel modes:

* Data Parallelism (DP for short)

* Model Parallelism (generally referred to as Tensor Parallelism, TP for short)

* Pipeline Parallelism (PP for short)

* Optimizer Parallelism (OP for short)

* Sequence Parallelism (SP for short)

* Multi-Copy Parallelism

In practice, multiple parallel strategies and multiple optimizations, such as using optimizer parallelism and recomputation, are usually employed to reduce the model's use of memory and improve training efficiency. Parallel strategy design is closely related to the efficiency of the model, and it is crucial to identify one or more sets of better parallel strategies before model tuning.

For details, refer to [Parallel Strategy Guide](https://www.mindspore.cn/mindformers/docs/en/dev/function/distributed_parallel.html).

### Recomputation

MindSpore uses automatic differentiation in backward mode to automatically derive the backward diagram based on the forward diagram computation flow, and the forward and backward diagrams together form a complete computation diagram. When computing some backward operators, the results of some forward operators need to be used, resulting in the need for the results to reside in memory. Until the backward operators that depend on them have been computed, the memory occupied by the results of these forward operators will not be reused. This phenomenon pushes up the memory spikes for training, and is particularly significant in large-scale network models.

To solve this problem, MindSpore provides the ability to recompute the forward operator without saving the results of the forward operator, so that this memory can be reused, and then recompute the forward operator when computing the backward operator, if the forward result is needed.

Re-computation is categorized in the following two ways:

* Full-recomputation

  For extreme environments where memory resources are extremely limited. In this mode, all activation values are recalculated when needed, except for saving the input data, minimizing the dependence on memory. However, the corresponding amount of computation increases significantly.

* Partial-recomputation

  This strategy preserves activation values that take up less memory space but are more expensive to recompute, such as Cast, SiLU-Mul. At the same time, activation recomputation is performed for activation values that occupy a large amount of memory but have relatively low recomputation costs. This method achieves efficient management of memory usage while ensuring model performance.

#### Cast Recomputation

RMSNorm generally uses high-precision (FP32) computation, and the input needs to be converted from low-precision (FP16 or BF16) to high-precision (FP32) via Cast before computation. RMSNorm needs to save the input for reverse computation. Therefore, recomputing Cast here only saves the low-precision input of Cast instead of the high-precision input of RMSNorm, a move that reduces the memory usage of that input by half, resulting in memory savings.

![cast](./images/cast.png)

Performing recomputation from high precision to low precision Cast operator will result in the later operators originally only need to store the low precision memory after Cast, and after the Cast operator recomputation, they need to store the high precision memory, which will result in larger memory usage instead.

#### SiLU-Mul Recomputation

In FeedForward, the middle part of the memory tends to be large. SiLU and Mul recomputation is less costly, so recomputing the SiLU and Mul operators saves memory for the first inputs of MatMul and Mul of w2.

![SiLU_mul](./images/silu_mul.png)

### Tools Introduction

#### profiler Tool

MindFormers itself integrates profiling data collection with the following steps:

1. Modify the configuration files

   Turn on the profiling switch in the model configuration file with the following parameters to be changed:

   ```yaml
   profile: True  # Whether to enable performance analysis tools
   profile_start_step: 5  # Step that starts performance analysis
   profile_stop_step: 6  # Step that ends performance analysis
   init_start_profile: False  # Enabled when Profiler is initialized, profile_start_step will not take effect after it is enabled.
   profile_communication: False # Whether to collect communication performance data in multi-NPU training
   profile_memory: True  # Collect Tensor memory data
   ```

   profile_start_step and profile_stop_step determine the collection interval, because the collection takes a long time. It is not recommended to set the interval too large, and it should be set to 2 to 4 steps. Since the first step involves compilation, it is recommended to start collecting from step 3.

2. View Data

   By default, the collection tool creates a `profile` folder under the `. /output` path, which can be set via the output_dir field of the model's yaml configuration file.

   The generated file and its introduction refer to [Introduction to profile file](https://www.mindspore.cn/docs/en/master/model_train/optimize/profiler.html), which mainly collects information such as running time of operators and tasks, CPU utilization and memory consumption for performance tuning analysis.

#### MindStudio Insight

MindStudio Insight provides multiple presentations of performance data, including visual presentations of Timeline views, communication analysis, computational elapsed time, so that users can analyze potential performance bottlenecks and provide guidance on how to take steps to eliminate or reduce them. MindStudio Insight supports viewing data exported by Profiling in Timeline View for cluster scenarios and displaying it in a single-card dimension, and can support cluster performance file analysis of more than 20GB.

Click [MindStudio Insight download link](https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann) and select the appropriate version to install.

Open MindStudio Insight, click the "+" in the toolbar at the top left of the interface, select the file or directory to be parsed and exported in the pop-up window, and then click “Confirm” to import.

MindStudio Insight tool presents the full process of online inference, training process in the form of a Timeline, and in accordance with the scheduling process to present the overall operating conditions, and the tool supports cluster Timeline display. By analyzing the timeline, users can analyze the online inference/training process at a fine-grained level, such as whether the iteration gap is too long, operator execution time, and provide easy-to-use features to assist users to quickly locate performance bottlenecks.

The Timeline interface consists of four parts: the toolbar (Area I), the timeline tree (Area II), the graphical pane (Area III), and the data pane (Area IV), as shown in the figure.

![studio](./images/studio.png)

* Area I

  The toolbar, which contains frequently used buttons, from left to right, is Marker List, Filter (supports filtering the display by card or by special layer), Search, Link Events, Recovery, Timeline Zoom Out and Timeline Zoom In.

* Area II

  Timeline tree diagram showing the hierarchical information of each “Card” in the cluster scenario, with “Card” at the first level, process or specialization hierarchies at the second level, and threads at the third level. This includes upper application data (containing elapsed time information of upper application arithmetic), CANN layer data (containing elapsed time data of AscendCL, GE, and Runtime components), underlying NPU data (containing elapsed time data and iteration trajectory data of each Stream task flow under Ascend Hardware, HCCL and Overlap Analysis communication data, and other Rise AI processor system data), hitpoint data, and the AI Core Freq hierarchy.

* Area III

  The graphical pane, which displays data within an iteration, corresponds to a timeline tree diagram, which provides a row-by-row graphical presentation of the timeline, including the execution sequence and execution duration of the upper-level application operators, components and interfaces.

* Area IV

  Data pane, statistical information or operator detail information display area, Slice Detail for detailed information on selected individual operators, Slice List for a list of operators in the selected area of a lane, and System View for a summary of operators in a category.

Click anywhere on the timeline page tree or graphical pane can be performed using the W (zoom in), A (move left), S (zoom out), and D (move right) keys in the keyboard, which support zooming in with a maximum precision of 1ns. This tool can provide overview, memory, arithmetic, communication and other dimensions of analysis to assist in performance tuning. Refer to [MindStudio Insight User Guide](https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/msinsightug/msascendinsightug/Insight_userguide_0002.html) for detailed usage.

#### IR Graph

In the [MindFormers configuration file](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html), just turn on save_graphs, and the runtime will output some intermediate files ending with the .ir suffix generated during the graph compilation process, which we call IR files. By default, a directory of graphs will be generated in the current task execution directory, and all IR graphs will be saved in this. It is a relatively intuitive and easy to understand document describing the structure of the model in text format, which can be viewed directly with text editing software. Refer to [Config Configuration Description](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html) for the meaning of the configuration items, and the configuration method is as follows:

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

  Reshape with the 14% output above to get <Tensor[Float16], (1, 4096, 4, 128)>.

* `%16`

  Transpose with the 15% output above to get <Tensor[Float16], (1, 4, 4096, 128)>.

It is recommended to change the number of layers of the model to a smaller size when saving IR graph, to reduce the time of compiling and saving graph, and to facilitate fast debugging. For details, please refer to [Introduction to IR file](https://www.mindspore.cn/docs/en/master/model_train/debug/error_analysis/mindir.html#ir-introduction) and [Analysis samples](https://www.mindspore.cn/docs/en/master/model_train/debug/error_analysis/mindir.html#how-to-derive-the-cause-of-the-failure-based-on-the-analyze-fail-ir-file-analysis-graph).

#### SAPP Automatic Load Balancing Tool

Large model training performance tuning requires simultaneous consideration of multi-dimensional hybrid parallel strategy configurations and memory constraints, and engineers need to try different combinations of schemes on the cluster to find a parallel strategy that achieves the required performance, and the process often takes weeks and consumes a lot of arithmetic costs.

MindSpore provides SAPP (Symbolic Automatic Parallel Planner) automatic load balancing tool. Inputting the model memory and time information, as well as some of the pipeline parallel performance-related hyper-references (e.g., the impact of recomputation on performance), the tool will construct the linear programming problem by itself, through the global solution, automatically generate stage-layer ratios in the pipeline parallel for the large model, adjust the recalculation strategy of each layer, automatically optimize the cluster arithmetic power and memory utilization, reduce the idle waiting time, realize the Pipeline parallel minute-level strategy optimization, greatly reduce the performance tuning cost, and significantly improve the end-to-end training performance.

For detailed usage, please refer to [SAPP Pipelined Load Balancing](https://gitee.com/mindspore/mindformers/tree/dev/toolkit/pipeline_balance) tool introduction.

## Overall Idea of Performance Tuning

### Overall Concept

The performance tuning of the large model mainly contains three parts of work: parallel strategy configuration, memory optimization, and time-consumption analysis. Performance optimization is a cyclic process, after the parallel strategy configuration is completed, it is necessary to carry out memory optimization analysis and memory optimization; and then conduct experimental analysis of the cluster distributed strategy to analyze whether the communication time consumption is reasonable and whether there is additional rearranging distribution overhead. Then according to the analysis results, adjust the parallel strategy, continue the memory, time-consumption analysis, the cycle of optimization, and then step by step to achieve the set performance goals.

After completing a round of performance optimization, it is also necessary to ensure that the model accuracy is aligned, and the alignment applies this optimization strategy.

### Parallel Strategy

#### Features of Parallel Strategy

The features of different parallel strategies are summarized below:

* Data parallelism

  Multiple pieces of data are trained at the same time and communicated only once at the gradient update, which is optimal performance without memory reduction.

* Model Parallelism

  Slicing the whole model into different Devices, the network computes the respective parts in parallel and communicates at locations such as LayerNorm, which saves the most memory but has a large amount of communication.

* Pipeline Parallelism

  Slices different stages of the model into different Devices, the network computes the respective stages serially and communicates when switching stages, saves some memory by recomputing, less communication, but there will be computational idleness (bubble).

* Optimizer parallelism

  Slicing the optimizer weights, model weights by DP (DP can exactly divide the 0th dimension of the weights shape) and communicating when the gradient is updated, which can save memory significantly and the communication is small.

* Sequence parallelism

  Short sequence parallelism slices the sequence by MP at LayerNorm, unchanged communication, reducing memory and some computation of Norm.

* Multi-copy parallelism

  In model parallelism, MatMul and other operators are sliced into multiple copies, and the computation and the communication between different copies are interleaved to achieve communication masking.

#### Suggestions

In practice, multiple parallelization strategies are usually used in combination. According to the model size, the number of machines to determine the appropriate parallelism strategy. This section describes the recommended configurations for different model sizes. The meanings of the configuration items in the sample configurations are referenced in [Config Configuration Description](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html)。

* Small parameter models

  When the model size is small (e.g. 7B), pure data parallelism + optimizer parallelism can be used, and gradient accumulation can be further turned on if memory is rich. Use 8-card training, [Llama2-7B parallelism strategy recommended configuration](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_7b.yaml).

* Moderate parameter models

  For moderate model sizes (e.g. 13B), further pipeline parallelism can be used and recomputation can be tuned. Use 8-card training, [Llama2-13B parallel strategy recommended configuration](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_13b.yaml).

* Large parameter models

  When the model size is large (e.g., 70B), model parallelism needs to be turned on, while sequence parallelism and multicopy parallelism are also recommended. Use 64-card training, [Llama2-70B parallel strategy recommended configuration](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_70b.yaml).

### Memory Optimization

During model training, computational resources are limited, and recomputation is required when memory is insufficient. Memory optimization mainly optimizes the configuration of recomputation, which can automatically generate the recommended recomputation configuration under the current parallel configuration with the help of the above SAPP tool.

MindSpore also provides DryRun functionality to simulate the memory consumption of each rank in a large cluster in a local environment for efficient device memory simulation without relying on actual large cluster resources.

After completing the recomputation configuration, first use DryRun to analyze, whether the required memory exceeds the maximum available memory, and if it does, the configuration needs to be readjusted. The maximum available memory is configured by the following field. The recommended value is `58G`, if it is set too large, it may cause other components to run out of memory.

```yaml
context:
  max_device_memory: "58GB"
```

Create a new script `dry_run.sh` with the following contents:

```shell
#!/bin/bash

YAML_FILE=$1
RANK_SIZE=$2
PIPELINE_STAGES=$3
RANK_GAP=$((RANK_SIZE/PIPELINE_STAGES))
ROOT_PATH=`pwd`

export MS_SIMULATION_LEVEL=1
export RANK_SIZE=$RANK_SIZE

rm -rf output_dryrun
mkdir output_dryrun
for((i=0; i<$PIPELINE_STAGES; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((i*RANK_GAP))
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    # The run_mindformer.py path needs to be specified correctly
    python ./run_mindformer.py --config $ROOT_PATH/$1 &> ./output_dryrun/rank_$RANK_ID.log &
done

```

Executing script

```shell
bash dry_run.sh $train.yaml $rank_size $stage
```

The meanings of the three parameters are as follows:

* $train.yaml: configuration file to be debugged
* $rank_size: the number of simulation cards
* $stage: the number of stages, equal to the number of pipeline parallels

After execution is complete, log messages for each stage are generated in the output directory `output_dryrun`, and the following message is printed at the end of each log.

```text
Device MOC memory size: 62432M
MindSpore Used memory size: 59392M
MindSpore memory base address: 0
Used peak memory usage (without fragments): 48874M
Actual peak memory usage (with fragments): 48874M
```

Used peak memory usage (without fragments): indicates the peak NPU memory usage without fragments, focus on this value and recommend not exceeding the maximum available memory.

Actual peak memory usage (with fragments): represents the peak NPU memory usage with fragments.

### Time-consumption Analysis

The two main components of time consumption are operator time consumption as well as communication time consumption, which relies on profiling data analysis, which is referenced in the above sections. Focus on analyzing the files ascend_timeline_display_0.json and rank-*_ascend_ms/ASCEND_PROFILER_OUTPUT/kernel_details.csv in the profiler folder of any rank.

Use the MindStudio Insight tool mentioned in the above section to parse ascend_timeline_display_0.json and statistically analyze whether the computation and communication time consumption is as expected. Then check kernel_details.csv to analyze the details of each operator.

### Typical Case

#### Silu-Mul Recomputation Not in Effect

Performing recomputation on Silu and Mul saves memory when fine-grained multicopy is on, but doing recomputation on Silu and Mul does not save memory when fine-grained multicopy is off. The localization process is as follows:

1. Confirmation that recomputation is configured

   Check if the Cast, Silu and Mul operators have the label "recompute: Bool(1)" in the IR graph. If they do, it means that the operators are equipped with recompute.

2. Checking for recomputation operators

   Check if the Cast, Silu and Mul operators have the label duplicated in IR graphs. The absence of labeled operators indicates that the actual computational graph does not recompute this part of the operator. Only Cast operator is with duplicated label in the following example.

   ```text
   %1834(CNode_108839) = PrimFunc_Cast(%1833, I64(43)) {instance name: cast} primitive_attrs: {output_names: [output], input_names: [x, dst_type], recompute: Bool(1)} cnode_attrs: {recompute_sub_graph: U64(64), recompute_id: I64(65), duplicated: Bool(1), need_cse_after_recompute: Bool(1)} cnode_primal_attrs: {micro: I64(0)}
       : (<Tensor[Float16], (1, 4096, 4096)>, <Int64, NoShape>) -> (<Tensor[Float32], (1, 4096, 4096)>)
   ```

3. Checking the reverse calculation input

   The inputs to the reverse operators of Silu and Mul are checked in the IR diagram to see if they are as expected, and there are Reshape operators between Silu and Mul, and between Mul and MatMul when fine-grained multicopy is off, and Silu, Mul, and MatMul are connected when fine-grained multicopy is on. The process is as follows:

![reshape](./images/reshape.png)

It can be seen that the cause is that the input shape of Linear in the fine-grained multicopy scenario is two-dimensional, while the input shape of Linear in the non-fine-grained multicopy scenario is three-dimensional, so a Reshape operator between Linear and Mul, and the lack of Reshape recalculation results in recalculation of Silu being optimized. The additional recalculation of the Reshape results in a normal memory reduction. The reference configuration is as follows:

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

After the final tuning, the Llama2-13B performance was optimized to 2562tokens/s/p, for a total improvement of 37%.
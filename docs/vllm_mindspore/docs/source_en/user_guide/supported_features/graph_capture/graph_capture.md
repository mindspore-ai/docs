# Entire Graph Capture and Playback

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/graph_capture/graph_capture.md)

The vLLM-MindSpore plugin can capture operators in a model and play back the operators when the same input shape is encountered, improving the operator delivery performance and reducing the bottleneck on the host. On the Ascend platform, ACLGraph is used to capture and play back graphs.

## Principles

ACLGraph graph capture and playback can be considered as graph construction through operator delivery. It consists of two main phases:

- **Graph capture**: This phase is the graph construction phase. The input with a fixed shape is required. During runtime, the operator delivered by each subgraph and the input and output addresses of each operator are recorded as the key data for subsequent playback.

- **Graph playback**: When the input of the previously captured shape is passed, the captured graph can be delivered and executed at a time, reducing the host overhead to be delivered.

Assume that it takes 20 μs to deliver an operator, and a network contains 100 operators. In normal cases, it takes 2000 μs (20 x 100) to deliver operators on the host. However, if the graph is captured, only an entire graph needs to be delivered. This is slightly slower than delivering a single operator, for example, 50 μs. In this case, the host time can be reduced by 40 times.

The ACLGraph playback requires that all the device memory addresses remain unchanged. Therefore, when ACLGraph is used for capture, the device memory address of the entire graph input needs to be reserved. As a result, the device memory for storing the input device memory of the corresponding shape needs to be reserved for each shape capture. Therefore, not all shapes can be captured. Instead, only small batches that are sensitive to host performance are captured and played back. In vLLM, the padding mechanism is used to ensure that all small batches can be captured. For example, if the maximum size of batches that can be captured is 64, the shapes of the following batch size are captured: 1, 2, 4, 6, 16, 24, 32, 40, 48, 56, and 64. That is, after the batch size is greater than 8, one graph is captured every eight batches. The graphs that are not directly captured are padded to the nearest number of batches that can be played back to ensure the performance of small batches.

### Capturing Sliced Graphs

ACLGraph does not support some specific operators and operations, which cannot be captured by ACLGraph. For example, stream synchronization, graphics memory allocation, host-side computation, and ACL operators (graphics memory is allocated during operator execution). Currently, the Attention operators (such as FlashAttention and PagedAttention) of MindSpore require tiling computation on the host. As a result, these Attention computations cannot be included in the ACLGraph and the model needs to be sliced into multiple subgraphs for capture based on the Attention operators.

vLLM natively calls the PyTorch graph slicing capability to slice the model into multiple subgraphs on the Python side for capture. However, MindSpore does not support graph slicing on the Python side. Instead, graphs are sliced and captured on the C++ backend. According to the Attention operator slicing, an *N*-layer model is usually sliced into *N*+1 subgraphs for capture. That is, there are *N*+1 subgraphs and *N* operators for delivery. Although the performance is not as good as that of the entire graph capture and playback, the host is not a performance bottleneck, and the host overhead can be effectively reduced.

## Enabling Entire-Graph Capture

vLLM natively supports graph capture and playback on the GPU platform using cudagraph. The vLLM-MindSpore plugin reuses the native management capability of vLLM, with the only difference being that cudagraph is replaced with ACLGraph. Therefore, the configuration for enabling graph capture is the same as that of vLLM, and is controlled by the build level. You can set `-O 3` in the startup command to enable ACLGraph capture.

```bash
vllm-mindspore serve -O 3 /path/to/Qwen2.5/model --trust-remote-code
```

Currently, the vLLM-MindSpore plugin captures and plays back only the entire graph of the model that is computed only by the decoder. This capability is not enabled in other inference phases. The vLLM-MindSpore plugin supports the following values for the **-O** option: **0**, **1**, **2**, and **3**. The graph capture capability is enabled only when the value is **3**. Other values have the same effect.

## Usage Limitations

### Graph Capture Limitations

Currently, each time the ACLGraph captures a graph, a stream is consumed. On the Ascend platform, the maximum number of streams is 2000. In addition, some operators (such as communication operators) also apply for streams. Therefore, the maximum number of graphs that can be captured by the ACLGraph is about 2000. To ensure that the feature does not conflict with other features and to ensure security, it is recommended that the number of subgraphs that can be captured be calculated based on 1800. Therefore, the shapes that can be captured are limited. The DeepSeek R1 model is used as an example. Generally, a 63-layer model is divided into 64 subgraphs, which can capture about 30 different shapes. According to the compute, the vLLM-MindSpore plugin captures a maximum batch size of 128 by default, that is, 19 different shapes. You can manually pass parameters to specify the shapes. The following example shows how to capture the shapes of 1, 2, 4, and 8 batches:

```bash
vllm-mindspore serve /path/to/Qwen2.5/model --compilation_config '{"level": "3", "capture_sizes": [1,2,4,8]}' --trust-remote-code
```

### Unsupported MLA

Currently, the vLLM-MindSpore plugin supports models of the PagedAttention type. For models that use the MLA operator, such as DeepSeek, the ACLGraph adaptation of MindSpore is faulty. Therefore, the feature of capturing the entire graph is not supported. You are advised not to enable this feature.

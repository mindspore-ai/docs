# init功能差异

PyTorch: 该接口支持的集合通信有3种：MPI，Gloo，NCCL。该接口在初始化backend的同时，还提供`world_size`、`rank`和`timeout`等内容的配置。

MindSpore：该接口当前仅支持2种集合通信：HCCL，NCCL。而`world_size`、`rank`和`timeout`等内容的配置并不在该接口中设置，调用该接口之前，需设置相应的环境变量。

# 安全

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/vllm_mindspore/docs/source_zh_cn/general/security.md)

通过vLLM-MindSpore插件在 Ascend 上使能推理服务时，由于服务化、节点通信、模型执行等必要功能需要使用一些网络端口，因此会存在一些安全问题。

## 服务化端口配置

使用vLLM-MindSpore插件启动推理服务时，需要相关 IP 与端口信息，包括：

1. `host`: 配置服务关联的 IP 地址，默认值为 `0.0.0.0`。
2. `port`: 配置服务关联的端口，默认值为 `8000`。
3. `data-parallel-address`: 配置数据并行管理 IP，默认值为 `127.0.0.1`。
    > 仅在使能了多节点数据并行的 vLLM 中生效。
4. `data-parallel-rpc-port`: 配置数据并行管理端口，默认值为 `29550`。
    > 仅在使能了多节点数据并行的 vLLM 中生效。

## 节点间通信

通过vLLM-MindSpore插件进行多节点部署时，使用默认配置进行节点间的通信是不安全的，包括以下场景：

1. MindSpore 分布式通信。
2. 模型TP、DP并行下的通信。

为了保证其安全性，应该部署在具有足够安全性的隔离网络环境中。

### vLLM 中关于节点间通信的可配置项

1. 环境变量
    * `VLLM_HOST_IP`: 可配置 vLLM 中进程间用于通信的 IP 地址。主要作用场景是传递给运行框架进行分布式通信组网。
    * `VLLM_DP_MASTER_IP`: 设置数据并行主节点 IP 地址（非服务化启动数据并行场景），默认值为 `127.0.0.1`。
    * `VLLM_DP_MASTER_PORT`: 设置数据并行主节点端口（非服务化启动数据并行场景），默认值为 `0`。
2. 数据并行配置项
    * `data_parallel_master_ip`: 设置数据并行时的主节点 IP 地址，默认值为 `127.0.0.1`。
    * `data_parallel_master_port`: 设置数据并行时的主节点端口，默认为 `29500`。

### 执行框架分布式通信

需要注意的是，vLLM-MindSpore插件 当前通过 MindSpore 进行分布式通信，与其相关的安全问题应该参考 [MindSpore官网](https://www.mindspore.cn/)。

## 安全建议

1. 网络隔离
    * 在隔离的专用网络上部署 vLLM 节点。
    * 通过网络分段防止未经授权的访问。
    * 设置恰当的防火墙规则。如：
        * 除了vLLM API服务监听的端口外，阻止所有其他连接。
        * 确保用于内部的通信端口仅被信任的主机或网络访问。
        * 不向公共互联网或不受信任的网络暴露内部端口。
2. 推荐配置行为
    * 总是配置相关参数，避免使用默认值，如通过 `VLLM_HOST_IP` 设置指定的 IP 地址。
    * 设定防火墙规则，只允许必要的端口有访问权限。
3. 管理访问权限
    * 在部署环境实施物理层和网络层的访问限制。
    * 管理好身份验证和授权。
    * 在所有系统组件的权限管理上，遵循最小权限原则。

## 参考

* [vLLM Security](https://docs.vllm.ai/en/stable/usage/security.html)
* [MindSpore](https://www.mindspore.cn/)

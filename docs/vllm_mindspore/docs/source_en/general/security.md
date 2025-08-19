# Security

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/general/security.md)

When enabling inference services using vLLM-MindSpore Plugin on Ascend, there may be some security-related issues due to the need for certain network ports for necessary functions such as serviceification, node communication, and model execution.

## Service Port Configuration

When starting the inference service using vLLM-MindSpore Plugin, relevant IP and port information is required, including:

1. `host`: Sets the IP address associated with the vLLM serve (default: `0.0.0.0`).
2. `port`: Sets the port for vLLM serve (default: `8000`).
3. `data-parallel-address`: Sets the IP address for DP (default: `127.0.0.1`).
    > Only for multi-node DP.
4. `data-parallel-rpc-port`: Sets the port for DP (default: `29550`).
    > Only for multi-node DP.

## Inter-Node Communication

All communications between nodes in a multi-node vLLM deployment are insecure by default. This includes:

1. MindSpore Distributed communications.
2. Tensor, Data parallel communication.

For security, it should be deployed in a sufficiently secure isolated network environment.

### Configuration Options for Inter-Node Communications

1. Environment Variables:
    * `VLLM_HOST_IP`: Sets the IP address for vLLM processes to communicate on, main scenario is to communicate in MindSpore distributed network.
    * `VLLM_DP_MASTER_IP`: Sets the IP address for data parallel(not for online-serving, default: `127.0.0.1`).
    * `VLLM_DP_MASTER_PORT`: Sets the port for data parallel(not for online-serving, default: `0`).
2. Data Parallel Configuration:
    * `data_parallel_master_ip`: Sets the IP address for data parallel(default: `127.0.0.1`).
    * `data_parallel_master_port`: Sets the port for data parallel(default: `29500`).

### Executing Framework Distributed Communication

It should be noted that vLLM-MindSpore Plugin use MindSpore's distributed communication. For detailed security information about MindSpore, please refer to the [MindSpore](https://www.mindspore.cn/en).

## Security Recommendations

1. Network Isolation:
    * Deploy vLLM nodes on a dedicated, isolated network.
    * Use network segmentation to prevent unauthorized access.
    * Implement appropriate firewall rules. Such as:
        * Block all incoming connections except to the TCP port the API server is listening on.
        * Ensure that ports used for internal communication are only accessible from trusted hosts or networks.
        * Never expose these internal ports to the public internet or untrusted networks.
2. Configuration Best Practices:
    * Always configure relevant parameters and avoid using default values, such as setting a specified IP address through `VLLM_HOST_IP`.
    * Configure firewalls to only allow necessary ports between nodes.
3. Access Control:
    * Restrict physical and network access to the deployment environment.
    * Implement proper authentication and authorization for management interfaces.
    * Follow the principle of least privilege for all system components.

## References

* [vLLM Security](https://docs.vllm.ai/en/stable/usage/security.html)
* [MindSpore](https://www.mindspore.cn/en)

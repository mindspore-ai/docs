# Deepseek r1 多节点 TP16 推理示例

如果一个节点环境无法支撑一个推理模型服务的运行，则考虑会使用多个节点资源运行推理模型。
VLLM 通过 Ray 对多个节点资源进行管理和运行。

以下将以 Deepseek r1 671B w8a8 为例，介绍双节点TP推理流程。

## 使用 docker 容器

在通过 Ray 执行多节点任务时，需要确保各个节点的执行配置都是一致的，包括但不限于：模型配置文件路径、Python环境等。
推荐通过 docker 镜像创建容器的方式屏蔽执行差异。镜像和创建容器方法如下：

```bash
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest

# 分别在主从节点新建docker容器
docker run -itd --name=mindspore_vllm --ipc=host --network=host --privileged=true \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/sbin/:/usr/local/sbin/ \
        -v /var/log/npu/slog/:/var/log/npu/slog \
        -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump \
        -v /var/log/npu/:/usr/slog \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/vnpu.cfg:/etc/vnpu.cfg \
        --shm-size="250g" \
        hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest \
        bash
```

## 下载模型权重

推理模型服务需要下再对应的模型配置与权重。

> 各节点的下载存放位置应该一致。

### Python 脚本工具下载

执行以下 Python 脚本，从[魔乐社区](https://modelers.cn)下载 MindSpore版本的DeepSeek-R1 W8A8权重及文件。其中`local_dir`由用户指定，请确保该路径下有足够的硬盘空间：

```python
from openmind_hub import snapshot_download
snapshot_download(repo_id="MindSpore-Lab/DeepSeek-R1-W8A8",
                  local_dir="/path/to/save/deepseek_r1_w8a8",
                  local_dir_use_symlinks=False)
```

### git-lfs 工具下载

执行以下代码，以确认`git-lfs`工具是否可用：

```bash
git lfs install
```

如果可用，将获得如下返回结果：

```text
Git LFS initialized.
```

不可用则需要先安装`git-lfs`，请参考[git-lfs](https://git-lfs.com)，或参考[faqs](../../../faqs/faqs.md)章节中关于`git-lfs安装`的阐述。
执行以下命令，下载权重：

```shell
git clone https://modelers.cn/MindSpore-Lab/DeepSeek-R1-W8A8.git
```

## 设置环境变量

分别在主从节点配置如下环境变量：

> 注：环境变量必须设置在 Ray 创建集群前，且当环境有变更时，需要通过 `ray stop` 将主从节点集群停止，并重新创建集群，否则环境变量将不生效。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_CUSTOM_PATH=$ASCEND_HOME_PATH/../
export GLOO_SOCKET_IFNAME=enp189s0f0 # ifconfig查找ip对应网卡的网卡名
export HCCL_SOCKET_IFNAME=enp189s0f0 # ifconfig查找ip对应网卡的网卡名
export TP_SOCKET_IFNAME=enp189s0f0 # ifconfig查找ip对应网卡的网卡名
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export vLLM_MODEL_BACKEND=MindFormers
export MINDFORMERS_MODEL_CONFIG=/path/to/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8.yaml
```

## 启动 Ray 进行多节点集群管理

在 Ascend 上，需要额外安装 pyACL 包来适配 Ray。且所有节点的 CANN 依赖版本需要保持一致。

### 安装 pyACL

pyACL (Python Ascend Computing Language) 通过 CPython 封装了 AscendCL 对应的 API 接口，使用接口可以管理 Ascend AI 处理器和对应的计算资源。

在对应环境中，获取相应版本的 Ascend-cann-nnrt 安装包后，解压出 pyACL 依赖包并单独安装，并将安装路径添加到环境变量中：

```shell
./Ascend-cann-nnrt_8.0.RC1_linux-aarch64.run --noexec --extract=./
cd ./run_package
./Ascend-pyACL_8.0.RC1_linux-aarch64.run --full --install-path=<install_path>
export PYTHONPATH=<install_path>/CANN-<VERSION>/python/site-packages/:\$PYTHONPATH
```

> 在 Ascend 的首页中可以下载 Ascend 运行包。如, 可以下载 [8.0.RC1.beta1](https://www.hiascend.cn/developer/download/community/result?module=cann&version=8.0.RC1.beta1) 对应版本的运行包。

### 多节点间集群

> 多节点集群管理前，需要检查各节点的 hostname 是否各异，如果存在相同的，需要通过 `hostname <new-host-name>` 设置不同的 hostname。

- 启动主节点 `ray start --head --port=<port-to-ray>`，启动成功后，会提示从节点的连接方式。如在 ip 为 `192.5.5.5` 的环境中，通过 `ray start --head --port=6379`，提示如下：

    ```text
    Local node IP: 192.5.5.5

    --------------------
    Ray runtime started.
    --------------------

    Next steps
      To add another node to this Ray cluster, run
        ray start --address='192.5.5.5:6379'

      To connect to this Ray cluster:
        import ray
        ray.init()

      To terminate the Ray runtime, run
        ray stop

      To view the status of the cluster, use
        ray status
    ```

- 从节点连接主节点 `ray start --address=<head_node_ip>:<port>`。
- 通过 `ray status` 查询集群状态，显示的NPU总数为节点总合，则表示集群成功。

   当有两个节点，每个节点有8个NPU时，其结果如下：

   ```shell
   ======== Autoscaler status: 2025-05-19 00:00:00.000000 ========
   Node status
   ---------------------------------------------------------------
   Active:
    1 node_efa0981305b1204810c3080c09898097099090f09ee909d0ae12545
    1 node_184f44c4790135907ab098897c878699d89098e879f2403bc990112
   Pending:
    (no pending nodes)
   Recent failures:
    (no failures)

   Resources
   ---------------------------------------------------------------
   Usage:
    0.0/384.0 CPU
    0.0/16.0 NPU
    0B/2.58TiB memory
    0B/372.56GiB object_store_memory

   Demands:
    (no resource demands)
   ```

## 启动 Deepseek r1 671B w8a8 模型在线服务

### 启动服务

vLLM MindSpore可使用OpenAI的API协议，部署为在线服务。以下是在线服务的拉起流程。

```bash
# 启动配置参数说明
vllm-mindspore serve
 --model=[模型Config/权重路径]
 --trust_remote_code # 使用本地下载的model文件
 --max-num-seqs [最大Batch数]
 --max_model_len [输出输出最大长度]
 --max-num-batched-tokens [单次迭代最大支持token数, 推荐4096]
 --block-size [Block Size 大小, 推荐128]
 --gpu-memory-utilization [显存利用率, 推荐0.9]
 --tensor-parallel-size [TP 并行数]
```

执行示例

```bash
# 主节点：
vllm-mindspore serve --model="/path/to/save/deepseek_r1_w8a8" --trust_remote_code --max-num-seqs=256 --max_model_len=32768  --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 16
```

### 发起请求

使用如下命令发送请求。其中`$PROMPT`为模型输入。

```bash
PROMPT="I am"
MAX_TOKEN=120
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/path/to/save/deepseek_r1_w8a8", "prompt": "$PROMPT", "max_tokens": $MAX_TOKEN, "temperature": 0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0}'
```

# Deepseek r1 DP EP推理示例

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/tutorials/deepseek_DP_EP/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md)

以下将以Deepseek r1 671B w8a8为例，介绍双机DP推理流程。

## 新建docker容器

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

### Python脚本工具下载

执行以下 Python 脚本，从[魔乐社区](https://modelers.cn)下载 MindSpore版本的DeepSeek-R1 W8A8权重及文件。其中`local_dir`由用户指定，请确保该路径下有足够的硬盘空间。

```python
from openmind_hub import snapshot_download
snapshot_download(repo_id="MindSpore-Lab/DeepSeek-R1-W8A8",
                  local_dir="/path/to/save/deepseek_r1_w8a8",
                  local_dir_use_symlinks=False)
```

### git-lfs工具下载

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

```bash
alias wget="wget --no-check-certificate"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_PATH=$ASCEND_HOME_PATH/../
export MINDFORMERS_MODEL_CONFIG=/usr/local/Python-3.11/lib/python3.11/site-packages/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8_ep4tp4.yaml # DP4 TP4 EP4
# export MINDFORMERS_MODEL_CONFIG=/usr/local/Python-3.11/lib/python3.11/site-packages/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8_ep16.yaml # DP16 EP16
export vLLM_MODEL_BACKEND=MindFormers
export GLOO_SOCKET_IFNAME=enp189s0f0 # ifconfig查找ip对应网卡的网卡名
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V1=1
export EXPERIMENTAL_KERNEL_LAUNCH_GROUP='thread_num:4,kernel_group_num:16'
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 启动Deepseek r1 671B w8a8模型在线服务

`vllm-mindspore`可使用OpenAI的API协议，部署为在线服务。以下是在线服务的拉起流程：

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
 --headless # 仅从节点需要配置，表示不需要服务侧相关内容
 --data-parallel-size [DP 并行数]
 --data-parallel-size-local [当前服务节点中的DP数，所有节点求和等于data-parallel-size]
 --data-parallel-start-rank [当前服务节点中负责的首个DP的偏移量]
 --data-parallel-address [主节点的通讯IP]
 --data-parallel-rpc-port [主节点的通讯端口]
 --enable-expert-parallel # 使能专家并行
```

执行示例：

```bash
# 主节点：
vllm-mindspore serve --model="/path/to/save/deepseek_r1_w8a8" --trust_remote_code --max-num-seqs=256 --max_model_len=32768  --max-num-batched-tokens=4096 --block-size=128  --gpu-memory-utilization=0.9 --tensor-parallel-size 4  --data-parallel-size 4 --data-parallel-size-local 2  --data-parallel-start-rank 0 --data-parallel-address 192.10.10.10  --data-parallel-rpc-port 12370 --enable-expert-parallel > log11  2>&1 &

# 从节点：
vllm-mindspore serve --model="/path/to/save/deepseek_r1_w8a8" --trust_remote_code --max-num-seqs=256 --max_model_len=32768  --max-num-batched-tokens=4096 --block-size=128  --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --headless  --data-parallel-size 4 --data-parallel-size-local 2  --data-parallel-start-rank 2 --data-parallel-address 192.10.10.10  --data-parallel-rpc-port 12370 --enable-expert-parallel  > log11  2>&1 &
```

## 发送请求

使用如下命令发送请求。其中`$PROMPT`为模型输入：

```bash
PROMPT="I am"
MAX_TOKEN=120
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/path/to/save/deepseek_r1_w8a8", "prompt": "$PROMPT", "max_tokens": $MAX_TOKEN, "temperature": 0}'
```

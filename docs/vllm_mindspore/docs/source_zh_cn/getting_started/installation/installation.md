# 安装指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/installation/installation.md)

本文档将介绍安装vLLM MindSpore环境的操作步骤。分为三种安装方式：

- [docker安装](#docker安装)：适合用户快速使用的场景；
- [pip安装](#pip安装)：适合用户需要指定安装版本的场景；
- [源码安装](#源码安装)：适合用户有增量开发vLLM MindSpore的场景。

## 版本配套

- OS：Linux-aarch64
- Python：3.9 / 3.10 / 3.11
- 软件版本配套

   | 软件 | 版本 | 对应分支 |
   | -----    | -----   |  ----- |
   |[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)     |   8.1      |  -    |
   |[MindSpore](https://www.mindspore.cn/install/) |  2.7    | master     |
   |[MSAdapter](https://git.openi.org.cn/OpenI/MSAdapter)| 0.2 | master  |
   |[MindSpore Transformers](https://gitee.com/mindspore/mindformers)|1.6      | dev |
   |[Golden Stick](https://gitee.com/mindspore/golden-stick)|1.1.0    | r1.1.0 |
   |[vLLM](https://github.com/vllm-project/vllm)      | 0.8.3 | v0.8.3   |
   |[vLLM MindSpore](https://gitee.com/mindspore/vllm-mindspore) | 0.2 | master  |

## 配置环境

在本章节中，我们将介绍[docker安装](#docker安装)、[pip安装](#pip安装)、[源码安装](#源码安装)三种安装方式，以及[快速验证](#快速验证)用例，用于验证安装是否成功。

### docker安装

在本章节中，我们推荐用docker创建的方式，以快速部署vLLM MindSpore环境，以下是部署docker的步骤介绍：

#### 拉取镜像

用户可执行以下命令，拉取vLLM MindSpore的docker镜像：

```bash
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest
```

拉取过程中，用户将看到docker镜像各layer的拉取进度。拉取成功后，用户可执行以下命令，确认docker镜像拉取成功：

```bash
docker images
```

#### 新建容器

用户在完成[拉取镜像](#拉取镜像)后，设置`DOCKER_NAME`与`IMAGE_NAME`以设置容器名与镜像名，并执行以下命令，以新建容器：

```bash
export DOCKER_NAME=vllm-mindspore-container  # your container name
export IMAGE_NAME=hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest  # your image name

docker run -itd --name=${DOCKER_NAME} --ipc=host --network=host --privileged=true \
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
        ${IMAGE_NAME} \
        bash
```

新建容器后成功后，将返回容器ID。用户可执行以下命令，确认容器是否创建成功：

```bash
docker ps
```

#### 进入容器

用户在完成[新建容器](#新建容器)后，使用已定义的环境变量`DOCKER_NAME`，启动并进入容器：

```bash
docker exec -it $DOCKER_NAME bash
```

### 源码安装

- **CANN安装**

    CANN安装方法与环境配套，请参考[CANN社区版软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)，若用户在安装CANN过程中遇到问题，可参考[昇腾常见问题](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/CANNFAQ/cannfaq_000.html)进行解决。

    CANN默认安装路径为`/usr/local/Ascend`。用户在安装CANN完毕后，使用如下命令，为CANN配置环境变量：

    ```bash
    LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package
    source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}/ascend-toolkit
    ```

- **vLLM前置依赖安装**

    vLLM的环境配置与安装方法，请参考[vLLM安装教程](https://docs.vllm.ai/en/v0.8.3/getting_started/installation/cpu.html)。其依赖`gcc/g++ >= 12.3.0`版本，可通过以下命令完成安装：

    ```bash
    yum install -y gcc gcc-c++
    ```

- **vLLM MindSpore安装**

    安装vLLM MindSpore，需要在拉取vLLM MindSpore源码后，执行以下命令，安装依赖包：

    ```bash
    git clone https://gitee.com/mindspore/vllm-mindspore.git
    cd vllm-mindspore
    bash install_depend_pkgs.sh
    ```

    编译安装vLLM MindSpore：

    ```bash
    pip install .
    ```

    上述命令执行完毕之后，将在`vllm-mindspore/install_depend_pkgs`目录下生成`mindformers`文件夹，将其加入到环境变量中：

    ```bash
    export PYTHONPATH=$MF_PATH:$PYTHONPATH
    ```

### 快速验证

用户可以创建一个简单的离线推理场景，验证安装是否成功。下面以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 为例。首先用户需要执行以下命令，设置环境变量：

```bash
export ASCEND_TOTAL_MEMORY_GB=64 # Please use `npu-smi info` to check the memory.
export vLLM_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
export vLLM_MODEL_MEMORY_USE_GB=32 # Memory reserved for model execution. Set according to the model's maximum usage, with the remaining environment used for kvcache allocation
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.
```

关于环境变量的具体含义，可参考[这里](../quick_start/quick_start.md#设置环境变量)。

用户可以使用如下Python脚本，进行模型的离线推理：

```python
import vllm_mindspore # Add this line on the top of script.
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "I am",
    "Today is",
    "Llama is"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Create a LLM
llm = LLM(model="Qwen2.5-7B-Instruct")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}. Generated text: {generated_text!r}")
```

若成功执行，则可以获得类似的执行结果：

```text
Prompt: 'I am'. Generated text: ' trying to create a virtual environment for my Python project, but I am encountering some'
Prompt: 'Today is'. Generated text: ' the 100th day of school. To celebrate, the teacher has'
Prompt: 'Llama is'. Generated text: ' a 100% natural, biodegradable, and compostable alternative'
```

用户也可以参考[快速开始](../quick_start/quick_start.md)章节，使用[在线服务](../quick_start/quick_start.md#在线服务)的方式进行验证。

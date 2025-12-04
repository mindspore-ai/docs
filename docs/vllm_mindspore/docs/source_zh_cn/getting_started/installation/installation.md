# 安装指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/installation/installation.md)

本文档将介绍vLLM-MindSpore插件的[版本配套](#版本配套)、安装步骤与[快速验证](#快速验证)用例，用于验证安装是否成功。其中安装步骤分为两种方式：

- [docker安装](#docker安装)：适合用户快速使用的场景；
- [源码安装](#源码安装)：适合用户有增量开发vLLM-MindSpore插件的场景。

## 版本配套

- OS：Linux-aarch64
- Python：3.9 / 3.10 / 3.11
- 依赖软件版本配套

   | 软件 | 配套版本与下载链接 |
   | -----    | -----   |
   | CANN  |   [8.3.RC1](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/index/index.html)      |
   | MindSpore |  [2.7.1](https://www.mindspore.cn/versions#2.7.1)    |
   | MSAdapter| [0.5.0](https://repo.mindspore.cn/mindspore/msadapter/version/202510/20251011/r0.3.0_20251011095813_951a8218d4c29785e48f304e720212b57056573e_newest/) |
   | MindSpore Transformers | [1.7.0](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0)  |
   | vLLM     | [0.9.1](https://repo.mindspore.cn/mirrors/vllm/version/202507/20250715/v0.9.1/) |

- vLLM-MindSpore插件代码仓与下载链接

   |代码仓链接 | 插件包下载链接 |
   | -----    | -----   |
   | [0.4.0](https://atomgit.com/mindspore/vllm-mindspore/tree/r0.4.0/) | [Python3.9](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/VllmMindSpore/ascend/aarch64/vllm_mindspore-0.4.0-cp39-cp39-linux_aarch64.whl)，[Python3.10](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/VllmMindSpore/ascend/aarch64/vllm_mindspore-0.4.0-cp310-cp310-linux_aarch64.whl)，[Python3.11](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/VllmMindSpore/ascend/aarch64/vllm_mindspore-0.4.0-cp311-cp311-linux_aarch64.whl) |

## docker安装

在本章节中，我们推荐使用docker创建的方式，快速部署vLLM-MindSpore插件环境。以下是部署docker的步骤介绍：

### 构建镜像

用户可执行以下命令，拉取vLLM-MindSpore插件代码仓库：

```bash
git clone https://atomgit.com/mindspore/vllm-mindspore.git
```

根据计算卡类型，构建镜像：

- 若为Atlas 800I A2，则执行

  ```bash
  bash build_image.sh
  ```

- 若为Atlas 300I Duo，则执行

  ```bash
  bash build_image.sh -a 310p
  ```

构建成功后，用户可以得到以下信息：

```text
Successfully built e40bcbeae9fc
Successfully tagged vllm_ms_20250726:latest
```

其中，`e40bcbeae9fc`为镜像id，`vllm_ms_20250726:latest`为镜像名与tag。用户可执行以下命令，确认docker镜像创建成功：

```bash
docker images
```

### 新建容器

用户在完成[构建镜像](#构建镜像)后，设置`DOCKER_NAME`与`IMAGE_NAME`以设置容器名与镜像名，并执行以下命令新建容器：

```bash
export DOCKER_NAME=vllm-mindspore-container  # your container name
export IMAGE_NAME=vllm_ms_20250726:latest  # your image name

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

新建容器成功后，将返回容器ID。用户可执行以下命令，确认容器是否创建成功：

```bash
docker ps
```

### 进入容器

用户在完成[新建容器](#新建容器)后，使用已定义的环境变量`DOCKER_NAME`，启动并进入容器：

```bash
docker exec -it $DOCKER_NAME bash
```

## 源码安装

### CANN安装

CANN安装方法与环境配套，请参考[CANN社区版软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)。若用户在安装CANN过程中遇到问题，可参考[昇腾常见问题](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/CANNFAQ/cannfaq_000.html)进行解决。

CANN默认安装路径为`/usr/local/Ascend`。用户在安装CANN完毕后，使用如下命令，为CANN配置环境变量：

```bash
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}/ascend-toolkit
```

### vLLM前置依赖安装

vLLM的环境配置与安装方法，请参考[vLLM安装教程](https://docs.vllm.ai/en/v0.9.1/getting_started/installation/cpu.html)。

### vLLM-MindSpore插件安装

vLLM-MindSpore插件有以下两种安装方式。**vLLM-MindSpore插件快速安装**适用于用户快速使用与部署的场景；**vLLM-MindSpore插件手动安装**适用于用户对组件有自定义修改的场景。

- **vLLM-MindSpore插件快速安装**

    采用快速安装脚本来安装vLLM-MindSpore插件，需要在拉取vLLM-MindSpore插件源码后，执行以下命令安装依赖包：

    ```bash
    git clone https://atomgit.com/mindspore/vllm-mindspore.git
    cd vllm-mindspore
    bash install_depend_pkgs.sh
    ```

    编译安装vLLM-MindSpore插件：

    ```bash
    pip install .
    ```

    也可以参考[软件配套](#版本配套)，根据对应的Python版本，下载vLLM-MindSpore插件包，进行pip安装。

- **vLLM-MindSpore插件手动安装**

    若用户对依赖的vLLM、MindSpore、MSAdapter等组件有自定义修改的需求，可以在本地准备好修改后的安装包，按照特定的顺序进行手动安装。安装顺序要求如下：

    1. 安装vLLM

        ```bash
        pip install /path/to/vllm-*.whl
        ```

    2. 安装MindSpore

        ```bash
        pip install /path/to/mindspore-*.whl
        ```

    3. 安装MindSpore Transformers

        ```bash
        pip install /path/to/mindformers-*.whl
        ```

    4. 安装MSAdapter

        ```bash
        pip install /path/to/msadapter-*.whl
        ```

    5. 安装vLLM-MindSpore插件

        需要先拉取vLLM-MindSpore插件源码，再执行安装：

        ```bash
        git clone https://atomgit.com/mindspore/vllm-mindspore.git
        cd vllm-mindspore
        pip install .
        ```

## 快速验证

用户可以创建一个简单的离线推理场景，验证安装是否成功。下面以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 为例。首先用户需要执行以下命令，设置环境变量：

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
```

关于环境变量的具体含义，可参考[环境变量](../quick_start/quick_start.md#设置环境变量)。

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

用户也可以参考[快速开始](../quick_start/quick_start.md)章节，使用[在线推理](../quick_start/quick_start.md#在线推理)的方式进行验证。

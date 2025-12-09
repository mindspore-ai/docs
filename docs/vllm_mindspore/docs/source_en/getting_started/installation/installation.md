# Installation Guide

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/installation/installation.md)

This document will introduce the [Version Matching](#version-compatibility) of vLLM-MindSpore Plugin, the installation steps for vLLM-MindSpore Plugin, and the [Quick Verification](#quick-verification) to verify whether the installation is successful. The installation steps provide two installation methods:

- [Docker Installation](#docker-installation): Suitable for quick deployment scenarios.
- [Source Code Installation](#source-code-installation): Suitable for incremental development of vLLM-MindSpore Plugin.

## Version Compatibility

- OS: Linux-aarch64
- Python: 3.9 / 3.10 / 3.11
- Depent Software version compatibility

   | Software | Version And Links |
   | -----    | -----   |
   | CANN  |   [8.3.RC1](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/index/index.html)      |
   | MindSpore |  [2.7.1](https://repo.mindspore.cn/mindspore/mindspore/version/202510/20251023/r2.7.1_20251023150800_71340184dc86527cb1ac22c992b9a9b240bcd366_newest/)    |
   | MSAdapter| [0.0.5](https://repo.mindspore.cn/mindspore/msadapter/version/202510/20251011/r0.3.0_20251011095813_951a8218d4c29785e48f304e720212b57056573e_newest/) |
   | MindSpore Transformers | [1.7.0](https://repo.mindspore.cn/mindspore/mindformers/version/202510/20251030/r1.7.0_20251030031507_8ccc49b3f6645d3d1abfab80b4c78f3cafe5c84e_newest/)  |
   | vLLM     | [0.11.0](https://repo.mindspore.cn/mirrors/vllm/version/202511/20251113/v0.11.0/) |
   | ms_custom_ops | [0.1.0](https://repo.mindspore.cn/mindspore/ms_custom_ops/version/202512/20251203/master_20251203031508_007121f1940bf26aa8c40d479eb6a56548897bf3_newest/) |
   | MindSpore ONE | [0.5.0](https://repo.mindspore.cn/mindspore-lab/mindone/version/202512/20251205/master_20251205093444_4be9653bfac58cedc70a5696b9b91f7d40e25ebb_newest/) |

## Docker Installation

We recommend using Docker for quick deployment of the vLLM-MindSpore Plugin environment. Below are the steps:

### Building the Image

User can execute the following commands to clone the vLLM-MindSpore Plugin code repository:

```bash
git clone https://gitee.com/mindspore/vllm-mindspore.git
```  

To build the image according to your npu type, follow these steps:

- For Atlas 800I A2:

  ```bash
  bash build_image.sh
  ```

- For Atlas 300I Duo:

  ```bash
  bash build_image.sh -a 310p
  ```

After a successful build, user will get the following output:

```text
Successfully built e40bcbeae9fc
Successfully tagged vllm_ms_20250726:latest
```

Here, `e40bcbeae9fc` is the image ID, and `vllm_ms_20250726:latest` is the image name and tag. User can run the following command to confirm that the Docker image has been successfully created:

```bash
docker images
```

### Creating a Container

After [building the image](#building-the-image), set `DOCKER_NAME` and `IMAGE_NAME` as the container and image names, then execute the following command to create the container:

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

The container ID will be returned if docker is created successfully. User can also check the container by executing the following command:

```bash
docker ps
```

### Entering the Container

After [creating the container](#creating-a-container), user can start and enter the container, using the environment variable `DOCKER_NAME`:

```bash
docker exec -it $DOCKER_NAME bash
```

## Source Code Installation

### CANN Installation

For CANN installation methods and environment configuration, please refer to [CANN Community Edition Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit). If you encounter any issues during CANN installation, please consult the [Ascend FAQ](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/CANNFAQ/cannfaq_000.html) for troubleshooting.

The default installation path for CANN is `/usr/local/Ascend`. After completing CANN installation, configure the environment variables with the following commands:

```bash
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}/ascend-toolkit
```

### vLLM Prerequisites Installation

For vLLM environment configuration and installation methods, please refer to the [vLLM Installation Guide](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/cpu.html).

### vLLM-MindSpore Plugin Installation

vLLM-MindSpore Plugin can be installed in the following two ways. **vLLM-MindSpore Plugin Quick Installation** is suitable for scenarios where users need quick deployment and usage. **vLLM-MindSpore Plugin Manual Installation** is suitable for scenarios where users require custom modifications to the components.

- **vLLM-MindSpore Plugin Quick Installation**

    To install vLLM-MindSpore Plugin, user needs to pull the vLLM-MindSpore Plugin source code and then runs the following command to install the dependencies:

    ```bash
    git clone https://gitee.com/mindspore/vllm-mindspore.git
    cd vllm-mindspore
    bash install_depend_pkgs.sh
    ```

    Compile and install vLLM-MindSpore Plugin:

    ```bash
    pip install .
    ```

    If pip version is greater than or equal to 25.3, users need to use the following command to compile and install vLLM-MindSpore Plugin:

    ```bash
    pip install --no-build-isolation .
    ```

    User can also refer to [Version Compatibility](#version-compatibility), check the Python version, download vLLM-Mindspore Pulgin whl package, and use pip to install.

- **vLLM-MindSpore Plugin Manual Installation**

    If users require custom modifications to dependent components such as vLLM, MindSpore, or MSAdapter, they can prepare the modified installation packages locally and perform manual installation in a specific sequence. The installation sequence requirements are as follows:

    1. Install vLLM

        ```bash
        pip install /path/to/vllm-*.whl
        ```

    2. Install MindSpore

        ```bash
        pip install /path/to/mindspore-*.whl
        ```

    3. Install MindSpore Transformers

        ```bash
        pip install /path/to/mindformers-*.whl
        ```

    4. Install MSAdapter

        ```bash
        pip install /path/to/msadapter-*.whl
        ```

    5. Install Custom Ops

        ```bash
        pip install /path/to/ms_custom_ops-*.whl
        ```

    6. Install MindSpore ONE

        ```bash
        pip install /path/to/mindone-*.whl
        ```

    7. Install vLLM-MindSpore Plugin

        User can use whl package to install vLLM-MindSpore Plugin.

        ```bash
        pip install /path/to/vllm_mindspore-*.whl
        ```

        User could also use source code to install vLLM-MindSpore Plugin.

        ```bash
        git clone https://gitee.com/mindspore/vllm-mindspore.git
        cd vllm-mindspore
        pip install .
        ```

        If pip version is greater than or equal to 25.3, users need to use the following command to compile and install vLLM-MindSpore Plugin:

        ```bash
        pip install --no-build-isolation .
        ```

## Quick Verification

User can verify the installation with a simple offline inference test. First, user needs to configure the environment variables with the following command:

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
```

About environment variables above, user can also refer to [environment variables section](../quick_start/quick_start.md#setting-environment-variables) for more details.

User can use the following Python scripts to verify with [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct):

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

If successful, the output will resemble:

```text
Prompt: 'I am'. Generated text: ' trying to create a virtual environment for my Python project, but I am encountering some'
Prompt: 'Today is'. Generated text: ' the 100th day of school. To celebrate, the teacher has'
Prompt: 'Llama is'. Generated text: ' a 100% natural, biodegradable, and compostable alternative'
```

Alternatively, refer to the [Quick Start](../quick_start/quick_start.md) guide for [online inference](../quick_start/quick_start.md#online-inference) verification.

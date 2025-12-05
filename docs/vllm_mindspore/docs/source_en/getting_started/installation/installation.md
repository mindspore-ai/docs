# Installation Guide

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/installation/installation.md)

This document will introduce the [Version Matching](#version-compatibility) of vLLM-MindSpore Plugin, the installation steps for vLLM-MindSpore Plugin, and the [Quick Verification](#quick-verification) to verify whether the installation is successful. The installation steps provide two installation methods:

- [Docker Installation](#docker-installation): Suitable for quick deployment scenarios.
- [Source Code Installation](#source-code-installation): Suitable for incremental development of vLLM-MindSpore Plugin.

## Version Compatibility

- OS: Linux-aarch64
- Python: 3.9 / 3.10 / 3.11
- Depent Software version compatibility

   | Software | Version And Links |
   | -----    | -----   |
   | CANN   |   [8.3.RC1](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/index/index.html) |
   | MindSpore  |  [2.7.1](https://www.mindspore.cn/versions#2.7.1) |
   | MSAdapter| [0.5.0](https://repo.mindspore.cn/mindspore/msadapter/version/202510/20251011/r0.3.0_20251011095813_951a8218d4c29785e48f304e720212b57056573e_newest/) |
   | MindSpore Transformers | [1.7.0](https://www.mindspore.cn/mindformers/docs/en/r1.7.0) |
   | vLLM       | [0.9.1](https://repo.mindspore.cn/mirrors/vllm/version/202507/20250715/v0.9.1/) |

- Source code and download link of vLLM-MindSpore Plugin

   | Source Code Link | Package Link |
   | -----    | -----   |
   | [0.4.0](https://atomgit.com/mindspore/vllm-mindspore/tree/r0.4.0/) | [Python3.9](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/VllmMindSpore/ascend/aarch64/vllm_mindspore-0.4.0-cp39-cp39-linux_aarch64.whl), [Python3.10](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/VllmMindSpore/ascend/aarch64/vllm_mindspore-0.4.0-cp310-cp310-linux_aarch64.whl), [Python3.11](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/VllmMindSpore/ascend/aarch64/vllm_mindspore-0.4.0-cp311-cp311-linux_aarch64.whl) |

## Docker Installation

We recommend using Docker for quick deployment of the vLLM-MindSpore Plugin environment. Below are the steps:

### Building the Image

User can execute the following commands to clone the vLLM-MindSpore Plugin code repository:

```bash
git clone https://atomgit.com/mindspore/vllm-mindspore.git
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

For vLLM environment configuration and installation methods, please refer to the [vLLM Installation Guide](https://docs.vllm.ai/en/v0.9.1/getting_started/installation/cpu.html).

### vLLM-MindSpore Plugin Installation

vLLM-MindSpore Plugin can be installed in the following two ways. **vLLM-MindSpore Plugin Quick Installation** is suitable for scenarios where users need quick deployment and usage. **vLLM-MindSpore Plugin Manual Installation** is suitable for scenarios where users require custom modifications to the components.

- **vLLM-MindSpore Plugin Quick Installation**

    To install vLLM-MindSpore Plugin, user needs to pull the vLLM-MindSpore Plugin source code and then runs the following command to install the dependencies:

    ```bash
    git clone https://atomgit.com/mindspore/vllm-mindspore.git
    cd vllm-mindspore
    bash install_depend_pkgs.sh
    ```

    Compile and install vLLM-MindSpore Plugin:

    ```bash
    pip install .
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

    5. Install vLLM-MindSpore Plugin

       User needs to pull source of vLLM-MindSpore Plugin, and run installation.

       ```bash
       git clone https://atomgit.com/mindspore/vllm-mindspore.git
       cd vllm-mindspore
       pip install .
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

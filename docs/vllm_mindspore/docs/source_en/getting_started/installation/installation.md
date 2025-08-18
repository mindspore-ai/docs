# Installation Guide

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/installation/installation.md)  

This document will introduce the [Version Matching](#version-compatibility) of vLLM-MindSpore Plugin, the installation steps for vLLM-MindSpore Plugin, and the [Quick Verification](#quick-verification) to verify whether the installation is successful. The installation steps provide two installation methods:

- [Docker Installation](#docker-installation): Suitable for quick deployment scenarios.
- [Source Code Installation](#source-code-installation): Suitable for incremental development of vLLM-MindSpore Plugin.  

## Version Compatibility

- OS: Linux-aarch64  
- Python: 3.9 / 3.10 / 3.11  
- Software version compatibility  

   | Software | Version And Links |
   | -----    | -----   |
   |[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)     |   [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)      |
   |[MindSpore](https://www.mindspore.cn/install/) |  [2.7.0](https://repo.mindspore.cn/mindspore/mindspore/version/202508/20250814/master_20250814091143_7548abc43af03319bfa528fc96d0ccd3917fcc9c_newest/unified/)    |
   |[MSAdapter](https://git.openi.org.cn/OpenI/MSAdapter)| [0.5.0](https://repo.mindspore.cn/mindspore/msadapter/version/202508/20250814/master_20250814010018_4615051c43eef898b6bbdc69768656493b5932f8_newest/any/) |
   |[MindSpore Transformers](https://gitee.com/mindspore/mindformers)| [1.6.0](https://gitee.com/mindspore/mindformers)  |
   |[Golden Stick](https://gitee.com/mindspore/golden-stick)| [1.2.0](https://repo.mindspore.cn/mindspore/golden-stick/version/202508/20250814/master_20250814010017_2713821db982330b3bcd6d84d85a3b337d555f27_newest/any/)  |
   |[vLLM](https://github.com/vllm-project/vllm)      | [0.9.1](https://repo.mindspore.cn/mirrors/vllm/version/202507/20250715/v0.9.1/any/) |
   |[vLLM-MindSpore Plugin](https://gitee.com/mindspore/vllm-mindspore) | [0.3.0](https://gitee.com/mindspore/vllm-mindspore/) |

## Docker Installation

We recommend using Docker for quick deployment of the vLLM-MindSpore Plugin environment. Below are the steps:  

### Building the Image  

User can execute the following commands to clone the vLLM-MindSpore Plugin code repository and build the image:

```bash  
git clone https://gitee.com/mindspore/vllm-mindspore.git
bash build_image.sh
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

For CANN installation methods and environment configuration, please refer to [CANN Community Edition Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=openEuler&Software=cannToolKit). If you encounter any issues during CANN installation, please consult the [Ascend FAQ](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/CANNFAQ/cannfaq_000.html) for troubleshooting.

The default installation path for CANN is `/usr/local/Ascend`. After completing CANN installation, configure the environment variables with the following commands:

```bash
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}/ascend-toolkit
```

### vLLM Prerequisites Installation

For vLLM environment configuration and installation methods, please refer to the [vLLM Installation Guide](https://docs.vllm.ai/en/v0.9.1/getting_started/installation/cpu.html). In vllM installation, `gcc/g++ >= 12.3.0` is required, and it could be  installed by the following command:

```bash
yum install -y gcc gcc-c++
```

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

    After executing the above commands, `mindformers` folder will be generated in the `vllm-mindspore/install_depend_pkgs` directory. Add this folder to the environment variables:  

    ```bash
    export PYTHONPATH=$MF_PATH:$PYTHONPATH  
    ```

- **vLLM-MindSpore Plugin Manual Installation**

    If user need to modify the components or use other versions, components need to be manually installed in a specific order. Version compatibility of vLLM-MindSpore Plugin can be found [Version Compatibility](#version-compatibility), abd vLLM-MindSpore Plugin requires the following installation sequence:  

    1. Install vLLM  

       ```bash  
       pip install /path/to/vllm-*.whl  
       ```  

    2. Uninstall Torch-related components  

       ```bash  
       pip uninstall torch torch-npu torchvision torchaudio -y  
       ```  

    3. Install MindSpore  

       ```bash  
       pip install /path/to/mindspore-*.whl  
       ```  

    4. Clone the MindSpore Transformers repository and add it to `PYTHONPATH`  

       ```bash  
       git clone https://gitee.com/mindspore/mindformers.git  
       export PYTHONPATH=$MF_PATH:$PYTHONPATH  
       ```  

    5. Install Golden Stick  

       ```bash  
       pip install /path/to/mindspore_gs-*.whl  
       ```  

    6. Install MSAdapter  

       ```bash  
       pip install /path/to/msadapter-*.whl  
       ```  

    7. Install vLLM-MindSpore Plugin

       User needs to pull source of vLLM-MindSpore Plugin, and run installation.

       ```bash  
       git clone https://gitee.com/mindspore/vllm-mindspore.git
       cd vllm-mindspore
       pip install .  
       ```

## Quick Verification

User can verify the installation with a simple offline inference test. First, user need to configure the environment variables with the following command:

```bash
export vLLM_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.
```

About environment variables above, user can also refer to [here](../quick_start/quick_start.md#setting-environment-variables) for more details.

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

Alternatively, refer to the [Quick Start](../quick_start/quick_start.md) guide for [online inference](../quick_start/quick_start.md#online-serving) verification.

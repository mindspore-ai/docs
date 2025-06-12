# Installation Guide  

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/installation/installation.md)  

This document describes the steps to install the vLLM MindSpore environment. Three installation methods are provided:  

- [Docker Installation](#docker-installation): Suitable for quick deployment scenarios.  
- [Pip Installation](#pip-installation): Suitable for scenarios requiring specific versions.  
- [Source Code Installation](#source-code-installation): Suitable for incremental development of vLLM MindSpore.  

## Version Compatibility

- OS: Linux-aarch64  
- Python: 3.9 / 3.10 / 3.11  
- Software version compatibility  

  | Software | Version | Corresponding Branch |  
  | -------- | ------- | -------------------- |  
  | [CANN](https://www.hiascend.com/developer/download/community/result?module=cann) | 8.1 | - |  
  | [MindSpore](https://www.mindspore.cn/install/) | 2.7 | master |  
  | [MSAdapter](https://git.openi.org.cn/OpenI/MSAdapter) | 0.2 | master |  
  | [MindSpore Transformers](https://gitee.com/mindspore/mindformers) | 1.6 | br_infer_deepseek_os |  
  | [Golden Stick](https://gitee.com/mindspore/golden-stick) | 1.1.0 | r1.1.0 |  
  | [vLLM](https://github.com/vllm-project/vllm) | 0.8.3 | v0.8.3 |  
  | [vLLM MindSpore](https://gitee.com/mindspore/vllm-mindspore) | 0.2 | master |  

## Environment Setup

This section introduces three installation methods: [Docker Installation](#docker-installation), [Pip Installation](#pip-installation), [Source Code Installation](#source-code-installation), and [Quick Verification](#quick-verification) example to check the installation.  

### Docker Installation

We recommend using Docker for quick deployment of the vLLM MindSpore environment. Below are the steps:  

#### Pulling the Image

Execute the following command to pull the vLLM MindSpore Docker image:  

```bash  
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest  
```  

During the pull process, user will see the progress of each layer. After successful completion, check the image by executing the following command:  

```bash  
docker images  
```  

#### Creating a Container

After [pulling the image](#pulling-the-image), set `DOCKER_NAME` and `IMAGE_NAME` as the container and image names, then execute the following command to create the container:  

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

The container ID will be returned if docker is created successfully. User can also check the container by executing the following command:  

```bash  
docker ps  
```  

#### Entering the Container

After [creating the container](#creating-a-container), user can start and enter the container, using the environment variable `DOCKER_NAME`:  

```bash  
docker exec -it $DOCKER_NAME bash  
```  

### Pip Installation

Use pip to install vLLM MindSpore, by executing the following command:  

```bash  
pip install vllm_mindspore  
```  

### Source Code Installation

- **CANN Installation**
  For CANN installation methods and environment configuration, please refer to [CANN Community Edition Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=openEuler&Software=cannToolKit). If you encounter any issues during CANN installation, please consult the [Ascend FAQ](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/CANNFAQ/cannfaq_000.html) for troubleshooting.

  The default installation path for CANN is `/usr/local/Ascend`. After completing CANN installation, configure the environment variables with the following commands:

  ```bash
  LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package
  source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
  export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}/ascend-toolkit
  ```

- **vLLM Prerequisites Installation**
  For vLLM environment configuration and installation methods, please refer to the [vLLM Installation Guide](https://docs.vllm.ai/en/v0.8.3/getting_started/installation/cpu.html). In vllM installation, `gcc/g++ >= 12.3.0` is required, and it could be  installed by the following command:

  ```bash
  yum install -y gcc gcc-c++
  ```

- **vLLM MindSpore Installation**  
  To install vLLM MindSpore, user needs to pull the vLLM MindSpore source code and then runs the following command to install the dependencies:

  ```bash  
  git clone https://gitee.com/mindspore/vllm-mindspore.git  
  cd vllm-mindspore  
  bash install_depend_pkgs.sh  
  ```  

  Compile and install vLLM MindSpore:  

  ```bash  
  pip install .  
  ```

  After executing the above commands, `mindformers-dev` folder will be generated in the `vllm-mindspore/install_depend_pkgs` directory. Add this folder to the environment variables:  

  ```bash  
  export MF_PATH=`pwd install_depend_pkgs/mindformers-dev`  
  export PYTHONPATH=$MF_PATH:$PYTHONPATH  
  ```  

  If MindSpore Transformers was compiled and installed from the `br_infer_deepseek_os` branch, `mindformers-os` folder will be generated in the `vllm-mindspore/install_depend_pkgs` directory. In this case, adjust the `MF_PATH` environment variable to:

  ```bash
  export MF_PATH=`pwd install_depend_pkgs/mindformers-os`
  export PYTHONPATH=$MF_PATH:$PYTHONPATH
  ```

### Quick Verification

To verify the installation, run a simple offline inference test with [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct):  

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

Alternatively, refer to the [Quick Start](../quick_start/quick_start.md) guide for [online serving](../quick_start/quick_start.md#online-serving) verification.

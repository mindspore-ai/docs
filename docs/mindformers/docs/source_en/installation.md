# Installation Guide

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/installation.md)

This guide is intended for the **PyNative (Dynamic Graph)** scenario and covers the environment dependencies, installation methods, and verification for MindSpore Transformers. Dynamic graph tasks are launched via `msrun ... run_mindformer.py --config xxx.yaml --mode 1` (`--mode 1` enables PyNative mode). Currently supported models include DeepSeek-V3 and others (see the [Model Library](introduction/models.md) for a complete list).

Installation consists of four main steps: **① Prepare Ascend hardware and install drivers/firmware → ② Install CANN and MindSpore according to the version compatibility table → ③ Install MindSpore Transformers → ④ Install HyperParallel (required for dynamic graph training)**. For step 3, you may choose any one of three methods: source, pip, or Docker.

> **Additional MindSpore Version Requirements for Dynamic Graph**
>
> The dynamic graph training stack depends on [HyperParallel](https://atomgit.com/mindspore/hyper-parallel/), which requires **MindSpore >= 2.10** (latest version recommended). Therefore, please select MindSpore >= 2.10. See the [Version Compatibility](#version-compatibility) section below for details.

## Environment Dependencies

### Ascend Hardware

Supported hardware includes Atlas 800T A2, Atlas 800I A2, and Atlas 900 A3 SuperPoD.

The host machine must have NPU drivers and firmware pre-installed. Refer to [Ascend Community - Install NPU Drivers and Firmware](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html).

### Python Version

| Item | Description |
|---|-----------------------------------------------------|
| **Supported Range** | Python `>= 3.12` |
| **Recommended Version** | Python `3.12.4`, consistent with the official Docker image's built-in `py3.12` and fully verified |

> "Supported Range" means the software can be installed and run normally; "Recommended Version" indicates the version on which official testing has been fully conducted. For new environments, it is recommended to use 3.12.4 directly to avoid compatibility issues with lower versions.

### Version Compatibility

Dynamic graph (r2.0.0 / WIP) requires the following components. Due to the dependency on HyperParallel, **MindSpore must be >= 2.10** (latest version recommended). CANN and drivers/firmware must correspond to the selected MindSpore version.

| Component | Version Requirement | Acquisition Method |
|---|---|---|
| MindSpore Transformers | WIP (master branch) | [Source Installation](#source-installation) |
| HyperParallel | WIP (matching MindSpore) | [Source Installation](#installing-hyperparallel-required-for-dynamic-graph-training) |
| MindSpore | **>= 2.10** (latest recommended) | [MindSpore Installation](https://www.mindspore.cn/install/) |

## Installing MindSpore Transformers

After completing CANN and MindSpore installation, proceed to install MindSpore Transformers. Choose one of the following three methods based on your scenario.

### Installation Method Selection

| Method | Use Case | Version Obtained |
|---|---|---|
| **Source Installation** | Need the latest master-branch features, want to modify source code, or debug | Current master branch code |
| **pip Installation** | Prefer to use a stable released version with an already-set-up environment | Published version on PyPI |
| **Docker Installation** | Prefer not to modify the host's Python/CANN/MindSpore environment and want a ready-to-use solution | Pre-installed released version in the image |

> The WIP (master) version currently **only supports source installation**; pip installation corresponds to published versions. Please select the version number according to the [Version Compatibility](#version-compatibility) table.

### Source Installation

```bash
# WIP version (master branch, latest features)
git clone -b master https://atomgit.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

```bash
# r2.0.0 version (r2.0.0 branch, corresponding to 2.0.0 release)
git clone -b r2.0.0 https://atomgit.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

The actual behavior of `build.sh` (see `build.sh` in the repository root directory):

1. Run `python setup.py bdist_wheel` to package the source code into `mindformers-*.whl` files in the `output/` directory;
2. Generate a `.sha256` checksum file for the resulting wheel;
3. **By default, it uses the Tsinghua mirror** `https://pypi.tuna.tsinghua.edu.cn/simple` to run `pip install mindformers*whl` and complete the installation.

> **Custom Installation Sources**
>
> `build.sh` uses the Tsinghua pip mirror by default. If you need to use a different source, first run `python setup.py bdist_wheel -d output` to generate the wheel, then install using your custom source:
>
> ```bash
> python setup.py bdist_wheel -d output
> pip install output/mindformers-*.whl -i <your pip source>
> ```

Compiling Prerequisites: Installing from source triggers local packaging. Please ensure that `wheel` and `setuptools` are installed, and that the dependencies in `requirements.txt` can be successfully fetched. If `bash build.sh` fails, first check: whether the pip repository is accessible via your network, whether your Python version is within the supported range, and whether you have successfully installed the matching version of MindSpore.

### pip Installation (Released Versions)

Released versions can be installed directly via pip. Refer to the [Version Compatibility](#version-compatibility) above for version numbers; ensure the version matches the entire row corresponding to your installed MindSpore:

```bash
# Install the specific version listed in the compatibility table (recommended to avoid mismatches)
pip install mindformers==2.0.0

# Or install the latest released version from PyPI
pip install mindformers
```

> Specifying a version number ensures compatibility with your installed CANN/MindSpore; omitting the version number will install the latest released version, in which case you must verify compatibility with your local MindSpore installation.
>
> `2.0.0` has not yet been released to PyPI, so installation via pip is currently unavailable. For now, please use the [“Install from Source”](#Install from Source) method described above to obtain the code from the master branch.

## Installing HyperParallel (Required for Dynamic Graph Training)

The dynamic graph (PyNative) training stack depends on [HyperParallel](https://atomgit.com/mindspore/hyper-parallel/), a distributed parallel acceleration library optimized for Ascend supernodes. It provides core capabilities for dynamic graphs, including `DTensor` / `DeviceMesh`, FSDP/HSDP, pipeline parallelism (PP, 1F1B/VPP) scheduling, and the MindSpore dynamic graph backward compatibility layer.

Trainers, optimizers, and parallel splitting and fusion operators under `mindformers/pynative/` all directly `import hyper_parallel`; therefore, **without HyperParallel installed, no dynamic graph tasks will run** (including single-card training).

> **Version Requirements**
>
> HyperParallel requires **MindSpore >= 2.10** (the latest version is recommended). Please ensure that MindSpore on your machine meets this requirement; see the installation instructions in the repository’s README for details.

The following demonstrates how to install HyperParallel from source:

```bash
git clone https://atomgit.com/mindspore/hyper-parallel.git
cd hyper-parallel
python setup.py bdist_wheel
pip install dist/hyper_parallel-*-py3-none-any.whl
```

After installation, verify that the import works correctly:

```bash
python -c “from hyper_parallel import DTensor, DeviceMesh, PipelineStage; print(‘hyper_parallel OK’)”
```

> **Docker Users**: When using the official pre-built image or building your own image, please verify that HyperParallel is already pre-installed in the image; if not, enter the container and install it using the source code method described above.

## Docker Installation

If you do not want to configure the Python/CANN/MindSpore environment directly on the host machine, you can run MindSpore Transformers using a Docker image. **We recommend using the official pre-built image first**; if you need a custom version combination, follow the instructions for building your own image.

### Environment and Tool Preparation

- **Hardware**: The host machine must have NPU drivers and firmware installed. Refer to [Ascend Community – Installing NPU Drivers and Firmware](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0005.html).
- **Software**: Docker version `26.1.4` or higher. You can verify this by running `docker --version`; if it is not installed, refer to the [Official Docker Installation Guide](https://docs.docker.com/engine/install/).
- **Network**: A stable internet connection is required to pull or build images, and access to Huawei Cloud is necessary. Please ensure that the host’s time and time zone are correct.

### Method 1: Use the Official Pre-built Image (Recommended)

The MindSpore Transformers Ascend image is hosted in the Huawei Cloud SWR image repository; it is ready to use out of the box and does not require local building.

> The official pre-built image for version 2.0 and later may not yet be available in the image repository; please refer to the actually available tags in [docker/OVERVIEW.md](https://atomgit.com/mindspore/mindformers/blob/master/docker/OVERVIEW.md). If the required version is not currently available, you can use `Method 2: Build the Image Yourself` described below.

**Image Repository URL:**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers
```

**Tag Specifications** (The system architecture is automatically recognized by the Docker Manifest; there is no need to specify it in the tag):

```text
<MindSpore Transformers version number>-<hardware information (chip)>-<operating system>-<Python version>
```

| Field | Example Value | Description |
|---|---|---|
| Version | `2.0.0` | MindSpore Transformers release version |
| Hardware Information (Chip) | `<Chip Architecture>` | Ascend chip model identifier; see [docker/OVERVIEW.md](https://atomgit.com/mindspore/mindformers/blob/master/docker/OVERVIEW.md) for specific values |
| Operating System | `ubuntu22.04` / `openeuler24.03` | Base image operating system distribution |
| Python Version | `py3.12` | Major Python version included in the image |

**Pull the image** (using `2.0.0-<chip architecture>-ubuntu22.04-py3.12` as an example):

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindformers:2.0.0-<chip architecture>-ubuntu22.04-py3.12
```

> Replace `<chip architecture>` with the actual chip model; see [docker/OVERVIEW.md](https://atomgit.com/mindspore/mindformers/blob/master/docker/OVERVIEW.md) for the complete list of supported architectures. You can use `docker manifest inspect <image>` to view the system architectures supported by the image (ARM64 / x86_64).

### Method 2: Build the Image Yourself

If you need a custom combination of CANN, MindSpore, and MindSpore Transformers versions, use the official Dockerfile in the repository (`docker/Dockerfile`) to build the image. This Dockerfile is based on the official CANN base image `quay.io/ascend/cann` and installs dependencies, MindSpore, and MindSpore Transformers in sequence.

```bash
docker build \
  --build-arg CANN_VERSION=9.1.0 \
  --build-arg CHIP_ARCH=<chip architecture> \
  --build-arg OS_SYSTEM=ubuntu22.04 \
  --build-arg PY_VERSION=py3.12 \
  --build-arg MINDSPORE_VERSION=2.10.0 \
  --build-arg MINDFORMERS_VERSION=2.0.0 \
  --build-arg PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
  -t mindformers:2.0.0-<chip architecture>-ubuntu22.04-py3.12 \
  -f docker/Dockerfile .
```

Meaning of each `build-arg` (consistent with the `docker/Dockerfile` repository):

| Parameter | Required | Description | Example Value |
|---|---|---|---|
| `CANN_VERSION` | Yes | Ascend CANN toolkit version (determines the base image tag) | `9.1.0` |
| `CHIP_ARCH` | Yes | Ascend chip architecture identifier; see [docker/OVERVIEW.md](https://atomgit.com/mindspore/mindformers/blob/master/docker/OVERVIEW.md) for specific values | `<chip architecture>` |
| `OS_SYSTEM` | Yes | Base image operating system and version | `ubuntu22.04` / `openeuler24.03` |
| `PY_VERSION` | Yes | Python version pre-installed in the base image | `py3.12` |
| `MINDSPORE_VERSION` | Yes | MindSpore version (select from the corresponding table) | `2.10.0` |
| `MINDFORMERS_VERSION` | Yes | MindSpore Transformers version | `2.0.0` |
| `PIP_INDEX_URL` | No | pip installation source URL; Huawei Cloud source by default | `https://mirrors.huaweicloud.com/repository/pypi/simple` |

> The version numbers listed above must comply with the [Version Compatibility](#version-compatibility). For more details, see the `docker/OVERVIEW.md` file in the repository.

### Start the Container

Once the pull or build is complete, mount the NPU devices and start the container:

```bash
docker run -itd \
  --ipc=host \
  --network=host \
  --device=/dev/davinci0:rwm \
  --device=/dev/davinci_manager:rwm \
  --device=/dev/devmm_svm:rwm \
  --device=/dev/hisi_hdc:rwm \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --name container_name \
  <image_name:tag> \
  /bin/bash
```

> **Security Note**
>
> - The example above mounts only one card, `davinci0`. Add `--device=/dev/davinciN:rwm` as needed to mount additional NPUs.
> - By default, the container runs as root. In production environments, it is recommended to create a non-privileged user to run the container.
> - To ensure NPU functionality, you may need to use `--privileged`, which expands the container’s privileges. It is recommended to use this only in trusted environments and to limit resources using `--cpus` and `--memory`.

## Verifying Successful Installation

After installation, run the following command to perform an environment self-check:

```bash
python -c “import mindformers as mf; mf.run_check()”
```

`run_check` collects environment information (MindFormers / MindSpore / CANN / driver versions) and returns one of three results based on this data.

### Result 1: Fully Passed

```text
- INFO - All checks passed, used **** seconds, the environment is correctly set up!
```

All component versions are compatible, and all checks have passed. You can now begin training.

### Result 2: Versions Do Not Match but All Checks Passed (Warning)

```text
- WARNING - The installed software versions do not match, but all checks passed in **** seconds
- INFO - It is recommended to install mindformers==x.y.z, mindspore==x.y.z, cann-toolkit==x.y.z, and driver==x.y.z
```

The functionality is available, but the current combination of MindFormers, MindSpore, CANN, and drivers is not among the recommended pairings. It is recommended to reinstall the entire suite using the recommended versions listed in the prompt and following the [Version Compatibility](#version-compatibility) to avoid potential accuracy, performance, or compatibility issues.

### Result 3: Check Failed

```text
- ERROR - The run check failed in **** seconds. It is recommended to install mindformers==x.y.z mindspore==x.y.z cann-toolkit==x.y.z driver==x.y.z
```

or

```text
- ERROR - The run check failed in **** seconds. Please check the information above for more details
```

Common Troubleshooting Steps:

| Symptom | Troubleshooting Steps |
|---|---|
| MindSpore reports an error (`MindSpore failed!`) | Refer to [MindSpore Installation](https://www.mindspore.cn/install/) and reinstall the matching version of MindSpore |
| Message indicating `ASCEND_HOME_PATH` is not set / CANN information not found | Check if CANN is installed and if the environment variables from `set_env.sh` have been `sourced` |
| Driver `version.info` not found | Check if the NPU driver/firmware is installed and if `npu-smi info` returns normal results |
| Version mismatch | Reinstall the entire version according to the recommended versions and [Version Compatibility](#version-compatibility) |
| Running a dynamic graph task results in `ModuleNotFoundError: hyper_parallel` | HyperParallel is not installed; see [Installing HyperParallel](#installing-hyperparallel-required-for-dynamic-graph-training) |

### Running a One-Step Dynamic Graph Self-Check

After `run_check` passes, it is recommended to use a minimal dynamic graph task to confirm that the end-to-end workflow can be launched. Start the dynamic graph task with `--mode 1` (PyNative) and launch it on a single machine with multiple GPUs using `msrun`, for example:

```bash
msrun --worker_num=8 --local_worker_num=8 --master_port=8118 \
  run_mindformer.py --config <your-configuration>.yaml --mode 1
```

For complete task launch steps and a minimal working example, see [Quick Start](quick_start/quick_start.md).

## Related Documentation

- First step after installation: [Quick Start](quick_start/quick_start.md) — Run your first dynamic graph task.
- Overall training process: [Training Guide](guide/training.md).
- List of supported models: [Model Library](introduction/models.md).
- Data and configuration: [Dataset](./feature/dataset.md), [Configuration File Documentation](./feature/configuration.md).
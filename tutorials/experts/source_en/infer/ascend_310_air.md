# Inference on the Ascend 310 AI Processor

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/infer/ascend_310_air.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Ascend 310 is a highly efficient and integrated AI processor oriented to edge scenarios. The Atlas 200 Developer Kit (Atlas 200 DK) is a developer board that uses the Atlas 200 AI accelerator module. Integrated with the HiSilicon Ascend 310 AI processor, the Atlas 200 allows data analysis, inference, and computing for various data such as images and videos, and can be widely used in scenarios such as intelligent surveillance, robots, drones, and video servers.

This tutorial describes how to use MindSpore to perform inference on the Atlas 200 DK based on the AIR model file. The process is as follows:

1. Prepare the development environment, including creating an SD card for the Atlas 200 DK, configuring the Python environment, and updating the development software package.

2. Export the AIR model file. The ResNet-50 model is used as an example.

3. Use the ATC tool to convert the AIR model file into an OM model.

4. Build the inference code to generate an executable `main` file.

5. Load the saved OM model, perform inference, and view the result.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/acl_resnet50_sample>.

## Preparing the Development Environment

### Hardware Preparation

- A server or PC with the Ubuntu OS is used to prepare a bootable SD card for the Atlas 200 DK and deploy the development environment.
- An SD card with a capacity of at least 16 GB.

### Software Package Preparation

The following five types of scripts and software packages are required for configuring the development environment:

1. Entry script for SD card preparation: [make_sd_card.py](https://gitee.com/ascend/tools/blob/master/makesd/for_1.0.9.alpha/make_sd_card.py).

2. Script for preparing a bootable SD card: [make_ubuntu_sd.sh](https://gitee.com/ascend/tools/blob/master/makesd/for_1.0.9.alpha/make_ubuntu_sd.sh).

3. Ubuntu OS image package: [ubuntu-18.04.xx-server-arm64.iso](http://cdimage.ubuntu.com/ubuntu/releases/18.04/release/ubuntu-18.04.6-server-arm64.iso). If the download is unsuccessful, try copying the link address and download it.

4. Driver package and running package of Atlas 200 DK:

    - `Ascend310-driver-*{software version}*-ubuntu18.04.aarch64-minirc.tar.gz`

    - `Ascend310-aicpu_kernels-*{software version}*-minirc.tar.gz`

    - `Ascend-acllib-*{software version}*-ubuntu18.04.aarch64-minirc.run`

5. Package for installing the development kit: `Ascend-Toolkit-*{version}*-arm64-linux_gcc7.3.0.run`

In the preceding information:

- For details about the first three items, see [Creating an SD Card with a Card Reader](https://support.huaweicloud.com/intl/en-us//usermanual-A200dk_3000/atlas200dk_02_0011.html).
- You are advised to obtain other software packages from [Firmware and Driver](https://ascend.huawei.com/en/#/hardware/firmware-drivers). On this page, select `Atlas 200 DK` from the product series and product model and select the required files to download.

### Preparing the SD Card

A card reader is connected to the Ubuntu server through a USB port, and the SD card is prepared using the script for SD card preparation. For details, see [Procedure](https://support.huaweicloud.com/intl/en-us/usermanual-A200dk_3000/atlas200dk_02_0011.html#section2).

### Connecting the Atlas 200 DK to the Ubuntu Server

The Atlas 200 DK can be connected to the Ubuntu server through a USB port or network cable. For details, see [Connecting the Atlas 200 DK to the Ubuntu Server](https://support.huaweicloud.com/intl/en-us/usermanual-A200dk_3000/atlas200dk_02_0013.html).

### Configuring the Python Environment

Install Python and GCC software. For details, see [Installing Dependencies](https://support.huaweicloud.com/intl/en-us/usermanual-A200dk_3000/atlas200dk_02_0016.html#section4).

### Installing the Development Kit

Install the development kit software package `Ascend-Toolkit-*{version}*-arm64-linux_gcc7.3.0.run`. For details, see [Installing the Development Kit](https://support.huaweicloud.com/intl/en-us/usermanual-A200dk_3000/atlas200dk_02_0017.html).

## Inference Directory Structure

Create a directory to store the inference code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/acl_resnet50_sample`. The `inc`, `src`, and `test_data` [sample code](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/acl_resnet50_sample) can be obtained from the official website, and the `model` directory stores the exported `AIR` model file and the converted `OM` model file. The `out` directory stores the executable file generated after building and the output result directory. The directory structure of the inference code project is as follows:

```text
└─acl_resnet50_sample
    ├── inc
    │   ├── model_process.h                   // Header file that declares functions related to resource initialization/destruction
    │   ├── sample_process.h                  // Header file that declares functions related to model processing
    │   ├── utils.h                           // Header file that declares common functions (such as the file reading function)
    ├── model
    │   ├── resnet50_export.air               // AIR model file
    │   ├── resnet50_export.om                // Converted OM model file
    ├── src
    │   ├── acl.json                          // Configuration file for system initialization
    │   ├── CMakeLists.txt                    // Build script
    │   ├── main.cpp                          // /Main function, which is the implementation file of image classification
    │   ├── model_process.cpp                 // Implementation file of model processing functions
    │   ├── sample_process.cpp                // Implementation file of functions related to resource initialization and destruction
    │   ├── utils.cpp                         // Implementation file of common functions (such as the file reading function)
    ├── test_data
    │   ├── test_data_1x3x224x224_1.bin       // Input sample data 1
    │   ├── test_data_1x3x224x224_2.bin       // input sample data 2
    ├── out
    │   ├── main                              // Executable file generated during building
    │   ├── result                            // Directory for storing the output result
```

> The output result directory `acl_resnet50_sample/out/result` must be created before inference.

## Exporting the AIR Model

Train the target network on the Ascend 910 AI Processor, save it as a checkpoint file, and export the model file in AIR format through the network and checkpoint file. For details about the export process, see [Export AIR Model](https://www.mindspore.cn/tutorials/en/master/advanced/train/save.html#export-air-model).

> The [resnet50_export.air](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com:443/sample_resources/acl_resnet50_sample/resnet50_export.air) is a sample AIR file exported using the ResNet-50 model.

## Converting the AIR Model File into an OM Model

Log in to the Atlas 200 DK environment, create the `model` directory for storing the AIR file `resnet50_export.air`, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/acl_resnet50_sample/model`, go to the directory, and set the following environment variables where `install_path` specifies the actual installation path:

```bash
export install_path=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages/te:${install_path}/atc/python/site-packages/topi:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

Take `resnet50_export.air` as an example. Run the following command to convert the model and generate the `resnet50_export.om` file in the current directory.

```bash
/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/atc/bin/atc --framework=1 --model=./resnet50_export.air --output=./resnet50_export --input_format=NCHW --soc_version=Ascend310
```

In the preceding information:

- `--model`: path of the original model file
- `--output`: path of the converted OM model file
- `--input_format`: input image format

For detailed information about ATC tools, please select the corresponding CANN version in the [Developer Documentation(Community Edition)](https://ascend.huawei.com/en/#/document?tag=developer), and then search for the chapter of "ATC Tool Instructions".

## Building Inference Code

Go to the project directory `acl_resnet50_sample` and set the following environment variables:

```bash
export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1
export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/lib64/stub/
```

> The `include` directory of the `acllib` package in the `CMakeLists.txt` file must be correctly specified. Otherwise, an error indicating that `acl/acl.h` cannot be found is reported. The code location of the `include` directory is as follows. If the location is inconsistent with the actual installation directory, modify it.

```text
...
#Header path

 include_directories(

  ${INC_PATH}/acllib_linux.arm64/include/

  ../

 )
...
```

Run the following command to create a build directory:

```bash
mkdir -p build/intermediates/minirc
```

Run the following command to switch to the build directory:

```bash
cd build/intermediates/minirc
```

Run the `cmake` command:

```bash
cmake ../../../src -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
```

Run the `make` command for building:

```bash
make
```

After building, the executable `main` file is generated in `acl_resnet50_sample/out`.

## Performing Inference and Viewing the Result

Copy the generated OM model file `resnet50_export.om` to the `acl_resnet50_sample/out` directory (the same path as the executable `main` file) and ensure that the input data sample is ready in the `acl_resnet50_sample/test_data` directory. Then, you can perform inference.

Note that the following environment variables must be set. Otherwise, the inference fails.

```bash
export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64/
```

Go to the `acl_resnet50_sample/out` directory. If the `result` directory does not exist in the current directory, run the `mkdir result` command to create one and run the following command to perform inference:

```bash
./main  ./resnet50_export.om  ../test_data
```

After the execution is successful, the following inference result is displayed. The `top5` probability label is displayed, and the output result is saved in the `.bin` file format in the `acl_resnet50_sample/out/result` directory.

```text
[INFO]  acl init success
[INFO]  open device 0 success
[INFO]  create context success
[INFO]  create stream success
[INFO]  get run mode success
[INFO]  load model ./resnet50_export.om success
[INFO]  create model description success
[INFO]  create model output success
[INFO]  start to process file:../test_data/test_data_1x3x224x224_1.bin
[INFO]  model execute success
[INFO]  top 1: index[2] value[0.941406]
[INFO]  top 2: index[3] value[0.291992]
[INFO]  top 3: index[1] value[0.067139]
[INFO]  top 4: index[0] value[0.013519]
[INFO]  top 5: index[4] value[-0.226685]
[INFO]  output data success
[INFO]  dump data success
[INFO]  start to process file:../test_data/test_data_1x3x224x224_2.bin
[INFO]  model execute success
[INFO]  top 1: index[2] value[0.946289]
[INFO]  top 2: index[3] value[0.296143]
[INFO]  top 3: index[1] value[0.072083]
[INFO]  top 4: index[0] value[0.014549]
[INFO]  top 5: index[4] value[-0.225098]
[INFO]  output data success
[INFO]  dump data success
[INFO]  unload model success, modelId is 1
[INFO]  execute sample success
[INFO]  end to destroy stream
[INFO]  end to destroy context
[INFO]  end to reset device is 0
[INFO]  end to finalize acl
```

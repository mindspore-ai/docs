# Environment Setup Guide

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/orange_pi/environment_setup.md)

This section describes how to burn an image on OrangePi AIpro, customize the installation of CANN and MindSpore, and configure the runtime environment.

## 1. Image Burning (Taking Windows as an example)

Image burning can be performed in any operating system. Here we will take Windows as an example to demonstrate how to quickly burn an image to your Micro SD card using the appropriate version of the balenaEtcher tool.

### 1.1 Preparation

Step 1 Insert the Micro SD card into the card reader and the card reader into the PC.

![environment-setup-1-1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-1.jpg)

### 1.2 Downloading the Ubuntu image

Step 1 Click [here](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro.html) to go to the mirror download page.

> This is only for illustration. Different power development board image download address is different, please check [here](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro(20T).html).

Step 2 Click the arrow icon in the picture to jump to the Baidu Wangpan download page.

![environment-setup-1-2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-2.png)

Step 3 Select the desktop version to download, it is recommended to download the 0318 version of the environment.

![environment-setup-1-3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-3.png)

Step 4 Alternative download method.

If the download from Baidu Wangpan is too slow, you can use [this link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/OrangePi/20240318/opiaipro_ubuntu22.04_desktop_aarch64_20240318.img.xz) to download directly.

### 1.3 Downloading the Tools

There are two card-making tools balenaEtcher, Rufus, and you can choose any one of the tools to burn according to your computer.

- balenaEtcher:

  Step 1 Download balenaEtcher.

  Click [here](https://etcher.balena.io/) to jump to the official website, and click the green download button to jump to where the software is downloaded.

  ![environment-setup-1-4](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-4.png)

  Step 2 Select to download the Portable version.

  The Portable version does not require installation, so double-click it to open it and use it.

  ![environment-setup-1-5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-5.png)

  Step 3 Alternative download method.

  If it is too slow to download from the official website, you can use to [this link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/OrangePi/balenaEtcher/balenaEtcher-Setup-1.18.4.exe) to download directly the balenaEtcher-Setup-1.18.4 software.

  Step 4  Open balenaEtcher.

  ![environment-setup-1-6](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-6.png)

  ![environment-setup-1-7](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-7.png)

- Rufus:

  Step 1 Download Rufus

  Click [this link](https://github.com/pbatard/rufus/releases/download/v4.5/rufus-4.5.exe) to download and install.

### 1.4 Selecting and Burning Images

Here we introduce balenaEtcher, Rufus to burn the image, you can burn according to the corresponding tool.

- balenaEtcher burns images:

  Step 1 Select Mirror, TF card, and start burn.

  1. Select the image file to be burned (the path where the Ubuntu image downloaded in 1.2 above is saved).

  2. Select the disk letter of the TF card.

  3. Click Start Burning, as shown below:

  ![environment-setup-1-8](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-8.png)

  It takes about 20 minutes to burn and verify, so please be patient:

  ![environment-setup-1-9](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-9.png)

  ![environment-setup-1-10](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-10.png)

  Step 2 Burning is complete.

  After the completion of burning, balenaEtcher is shown in the following figure, if the green indicator icon shows that the image is burned successfully, at this time you can exit balenaEtcher, pull out the TF card and insert it into the TF card slot on the development board to use:

  ![environment-setup-1-11](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-11.png)

- Rufus burns images:

  Step 1 Select Mirror, TF card, and start burn.

  Insert the sd card into the card reader, insert the card reader into the computer, select the image and sd card, click “Start”.

  ![environment-setup-1-12](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-12.png)

  Step 2 Burning is complete.

  Pull out the card reader directly after the wait is over.

  ![environment-setup-1-13](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-13.png)

## 2. CANN Upgrading

### 2.1 Toolkit Upgrading

Step 1 Open a terminal and switch the root user.

Use `CTRL+ALT+T` or click on the icon with `$_` at the bottom of the page to open the terminal.

![environment-setup-1-14](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-14.png)

Switch the root user, root user password: Mind@123.

```bash

# Open a terminal on the development board and run the following command

(base) HwHiAiUser@orangepiaipro:~$ su – root
 Password:
(base) root@orangepiaipro: ~#

```

Step 2 Remove installed CANN packages from the image to free up disk space and prevent installing new CANN packages from reporting low disk space errors.

```bash

(base) root@orangepiaipro: ~# cd /usr/local/Ascend/ascend-toolkit
(base) root@orangepiaipro: /usr/local/Ascend/ascend-toolkit # rm -rf *

```

Step 3 Open the official website of Ascend CANN to access the community version of the resource [download address](https://www.hiascend.com/developer/download/community/result?module=cann), download the required version of the toolkit package. Taking 8.0.RC2.alpha003 version as an example, as shown below:

![environment-setup-1-15](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-15.png)

> Execute the following commands to select the aarch64 or x86_64 package according to the actual output of the environment.

   ```bash
  uname -a
   ```

Step 4 Go to the Toolkit package download directory.

```bash
(base) root@orangepiaipro: /usr/local/Ascend/ascend-toolkit# cd /home/HwHiAiUser/Downloads
```

> Orange Pi AI Pro browser file default download directory: /home/HwHiAiUser/Downloads, users should synchronize to modify the path in the above command when changing the save path.

Step 5 Add execution permissions to the CANN package.

```bash
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# chmod +x ./Ascend-cann-toolkit_8.0.RC2.alpha003_linux-aarch64.run
```

Step 6 Execute the following command to upgrade the software.

```bash
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads#./Ascend-cann-toolkit_8.0.RC2.alpha003_linux-aarch64.run --install
```

Type Y when this prompt pops up during installation, then press Enter to continue the installation.

![environment-setup-1-16](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-16.png)

After the upgrade is completed, if the following message is displayed, the software upgrade is successful:

```bash
xxx install success

```

- xxx indicates the actual package name of the upgrade.

- Path after installing the upgrade (default installation path for root user as an example): “/usr/local/Ascend/ ascend-toolkit/

Step 7 Configure and load environment variables.

```bash

(base) root@orangepiaipro: /home/HwHiAiUser/Downloads # echo “source /usr/local/Ascend/ascend-toolkit/set_env.sh” >> ~/.bashrc
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads # source ~/.bashrc

```

### 2.2 Kernels Upgrading

> The binary arithmetic package Kernels relies on the CANN package Toolkit. To perform the upgrade, the current environment needs to have the matching version of Toolkit installed and installed by the same user.

Step 1 Open a terminal and switch the root user.

Password for root user: Mind@123.

```bash

# Open a terminal on the development board and run the following command

(base) HwHiAiUser@orangepiaipro:~$ su – root
 Password:
(base) root@orangepiaipro: ~#

```

Step 2 Execute the following command to get the development board NPU model number.

```bash
npu-smi info
```

Step 3 Open the official website of Ascend CANN to access the community edition resources [download address](https://www.hiascend.com/developer/download/community/result?module=cann), and download the kernel package that is consistent with the CANN package version and matches the NPU model. As shown in the figure below:

![environment-setup-1-18](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-18.png)

Step 4 Go to the Kernels package download directory.

```bash
(base) root@orangepiaipro: /usr/local/Ascend/ascend-toolkit# cd /home/HwHiAiUser/Downloads
```

> Default download directory of Orange Pi AI Pro browser file: /home/HwHiAiUser/Downloads

Step 5 Add execution permissions to the kernels package.

```bash
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads# chmod +x ./Ascend-cann-kernels-310b_8.0.RC2.alpha003_linux.run
```

Step 6 Execute the following command to upgrade the software.

```bash
(base) root@orangepiaipro: /home/HwHiAiUser/Downloads#./Ascend-cann-kernels-310b_8.0.RC2.alpha003_linux.run --install
```

After the upgrade is completed, if the following message is displayed, the software upgrade is successful:

```bash
xxx install success
```

- xxx indicates the actual package name of the upgrade.

- Path after installing the upgrade (default installation path for root user as an example): "/usr/local/Ascend/ ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel".

## 3. MindSpore Upgrading

### 3.1 Installing the Official Version of the Website (Taking MindSpore 2.3.1 as an example)

Method 1: Open the terminal as HwHiAiUser user and run the pip install command directly in the terminal.

```bash
pip install mindspore==2.3.1
```

Method 2: Refer to [MindSpore official website installation tutorial](https://www.mindspore.cn/install/en) to install.

```bash

pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.1/MindSpore/unified/aarch64/mindspore-2.3.1-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# Confirm the operating system and programming language, and the default environment of the Orange Pi development board is linux-aarch64 and python3.9

```

### 3.2 Installing the MindSpore daily Package (Using the September 11 daily Package as an Example)

The Orange Pi development board supports custom installation of MindSpore daily packages, which can be obtained from [this link](https://repo.mindspore.cn/mindspore/mindspore/version/) for the corresponding date.

- Specific lookup procedure of target daily whl package is as follows:

  1. Enter the directory prefixed with master. If there are multiple directories prefixed with master, it is recommended that you enter a directory with a later date.

  2. Enter the unified directory.

  3. According to the actual operating system information, enter the corresponding directory. Since the default operating system of Orange Pi board is linux-aarch64, enter the aarch64 directory. 4.

  4. According to the actual python version information, find the corresponding daily whl package. Since the default Orange Pi board is python 3.9, the target daily package is mindspore-2.4.0-cp39-cp39-linux_aarch64.whl.

  ![environment-setup-1-19](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/orange_pi/images/environment_setup_1-19.png)

> This tutorial aims to let developers experience the latest version features, but because the daily package is not the official release version, there may be some problems during operation. Developers can submit issues through the [community](https://gitee.com/mindspore/mindspore), or can be modified and submit their own PR.

- Download the whl package for installation and run the following command in the terminal.

```bash
# wget download whl package
wget https://repo.mindspore.cn/mindspore/mindspore/version/202409/20240911/master_20240911160029_917adc670d5f93049d35d6c3ab4ac6aa2339a74b_newest/unified/aarch64/mindspore-2.4.0-cp39-cp39-linux_aarch64.whl

# Go to the path of the whl package in the terminal and run the pip install command to install it.
pip install mindspore-2.4.0-cp39-cp39-linux_aarch64.whl
```
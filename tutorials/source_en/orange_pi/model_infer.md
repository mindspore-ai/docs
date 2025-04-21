# Model Online Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/orange_pi/model_infer.md)

This section describes how to download the Ascend MindSpore online inference case on the OrangePi AIpro (hereafter: OrangePi development board) and launch the Jupyter Lab interface to perform inference.

> All the following operations are performed under the HwHiAiUser user.

## 1. Downloading Case

Step1 Use the `CTRL+ALT+T` shortcut key or click on the icon with `$_` at the bottom of the page to open the terminal and download the case code.

```bash
# Open a terminal on the development board and run the following command
(base) HwHiAiUser@orangepiaipro:~$ cd samples/notebooks/
(base) HwHiAiUser@orangepiaipro:~$ git clone https://github.com/mindspore-courses/orange-pi-mindspore.git
```

Step2 Enter the case catalog.

The downloaded code package is in the following directory of the OrangePi AIpro development board: /home/HwHiAiUser/samples/notebooks.

The project catalog is listed below:

```bash
/home/HwHiAiUser/samples/notebooks/orange-pi-mindspore/Online/
01-quick_start
02-ResNet50
03-ViT
04-FCN
05-Shufflenet
06-SSD
07-RNN
08-LSTM+CRF
09-GAN
10-DCGAN
11-Pix2Pix
12-Diffusion  
13-ResNet50_transfer
14-qwen1.5-0.5b
15-tinyllama
16-DctNet
17-DeepSeek-R1-Distill-Qwen-1.5B
```

## 2. Inference Execution

Step 1 Launch the Jupyter Lab interface.

```bash
(base) HwHiAiUser@orangepiaipro:~$ cd /home/HwHiAiUser/samples/notebooks/  
(base) HwHiAiUser@orangepiaipro:~$ ./start_notebook.sh
```

After executing the script, the following printout will appear in the terminal, in which there will be a link to the URL for logging into Jupyter Lab.

![model-infer1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/orange_pi/images/model_infer1.png)

Then open the browser.

![model-infer2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/orange_pi/images/model_infer2.png)

Then enter the URL link you see above in your browser to log into the Jupyter Lab software.

![model-infer3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/orange_pi/images/model_infer3.png)

Step 2 In the Jupyter Lab interface, double-click the case directory shown in the figure below, take “04-FCN” as an example here, you can enter the case directory. The operation process of other cases is similar, just select the corresponding case directory and. ipynb file.

![model-infer4](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/orange_pi/images/model_infer4.png)

Step 3 In this directory there are all the resources to run the sample, where mindspore_fcn8s.ipynb is the file to run the sample in Jupyter Lab. Double-click to open the mindspore_fcn8s.ipynb, which will be displayed in the right window. The contents of the mindspore_fcn8s.ipynb file is shown in the following figure:

![model-infer5](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/orange_pi/images/model_infer5.png)

Step 4 Click the ⏩ button to run the sample. In the pop-up dialog box, click the "Restart" button, then the sample begins to run.

![model-infer6](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/tutorials/source_zh_cn/orange_pi/images/model_infer6.png)

## 3. Environment Cleaning

After the inference execution is completed, it is necessary to navigate to `KERNELS > Shut Down All` in the Jupyter Lab interface and manually close the already running kernel to prevent running other cases from reporting an error of insufficient memory, which may cause other cases to fail to execute properly.

![model-infer7](../../source_zh_cn/orange_pi/images/model_infer7.png)

## Next Suggestion

For specific case development based on MindSpore, please refer to [Quick Start](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/orange_pi/dev_start.html)

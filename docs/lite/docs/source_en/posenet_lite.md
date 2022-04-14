# Posenet Model

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/docs/source_en/posenet_lite.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## Posenet introduction

Under the detection of photo cameras, posenet model can identify the facial features and body posture of the human body in different positions.

The output of using the bone detection model is as follows:

The blue marking points detect the distribution of facial features of the human body and the skeletal trend of upper and lower limbs. During this infernece, the probability score is 0.98/1, and the inference time is 66.77ms.

![image_posenet](images/posenet_detection.png)

Using MindSpore Lite to realize posenet [example](https://gitee.com/mindspore/models/tree/r1.7/official/lite/posenet).

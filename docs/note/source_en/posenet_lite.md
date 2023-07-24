# Posenet Model Support (Lite)

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/note/source_en/posenet_lite.md)

## Posenet introduction

Under the detection of photo cameras, posenet model can identify the facial features and body posture of the human body in different positions.

The output of using the bone detection model is as follows:

The blue marking points detect the distribution of facial features of the human body and the skeletal trend of upper and lower limbs. During this infernece, the probability score is 0.98/1, and the inference time is 66.77ms.

![image_posenet](images/posenet_detection.png)

Using MindSpore Lite to realize posenet [example](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/lite/posenet).

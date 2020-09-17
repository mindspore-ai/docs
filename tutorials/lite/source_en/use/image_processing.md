# Preprocess image data

<!-- TOC -->

- [Preprocess image data](#preprocess-image-data)
    - [Overview](#Overview)
    - [Import image preprocessing function library](#import-image-preprocessing-function-library)
    - [Initialize the image](#initialize-the-image)
        - [Usage example](#usage-example)
    - [Optional image preprocessing operator](#optional-image-preprocessing-operator) 
        - [Resize image](#resize-image)
            - [Usage example](#usage-example-1) 
        - [Convert the image data type](#convert-the-image-data-type)
            - [Usage example](#usage-example-2) 
        - [Crop image data](#crop-image-data)
            - [Usage example](#usage-example-3) 
        - [Normalize image data](#normalize-image-data)
            - [Usage example](#usage-example-4)        

<!-- /TOC -->

## Overview

The main purpose of image preprocessing is to eliminate irrelevant information in the image, restore useful real information, enhance the detectability of related information and simplify data to the greatest extent, thereby improving the reliability of feature extraction, image segmentation, matching and recognition. Here, by creating a LiteMat object, the image data is processed before inference to meet the data format requirements for model inference.

The process is as follows:

## Import image preprocessing function library

```
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"
```

## Initialize the image

Here, the [InitFromPixel](https://www.mindspore.cn/lite/docs/en/master/apicc/dataset.html#initfrompixel) function in the `image_process.h` file is used to initialize the image.

```
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m);
```

### Usage example

```
// Create the data object of the LiteMat object.
LiteMat lite_mat_bgr;

// Initialize the lite_mat_bgr object.
// The image data pointer passed in by the user (The data in the Bitmap corresponding to the Android platform).
InitFromPixel(pixel_ptr, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
```

## Optional image preprocessing operator

The image processing operators here can be used in any combination according to the actual situation.

### Resize image

Here we use the [ResizeBilinear](https://www.mindspore.cn/lite/docs/en/master/apicc/dataset.html#resizebilinear) function in `image_process.h` to resize the image through a bilinear algorithm. Currently, the supported data type is unit8, the supported channels are 3 and 1.

```
bool ResizeBilinear(const LiteMat &src, LiteMat &dst, int dst_w, int dst_h);
```

#### Usage example

```
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// Create a resize image data object.
LiteMat lite_mat_resize;

// Resize the image.
ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
```

### Convert the image data type

Here we use the [ConvertTo](https://www.mindspore.cn/lite/docs/en/master/apicc/dataset.html#convertto) function in `image_process.h` to convert the image data type. Currently, the supported conversion is to convert uint8 to float.

```
bool ConvertTo(const LiteMat &src, LiteMat &dst, double scale = 1.0);
```

#### Usage example

```
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// Create the converted data type object.
LiteMat lite_mat_convert_float;

// Perform conversion type operations on the object. Currently, the supported conversion is to convert uint8 to float.
ConvertTo(lite_mat_bgr, lite_mat_convert_float);
```

### Crop image data

Here we use the [Crop](https://www.mindspore.cn/lite/docs/en/master/apicc/dataset.html#crop) function in `image_process.h` to crop the image. Currently, channels 3 and 1 are supported.

```
bool Crop(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h);
```

#### Usage example

```
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// Create the cropped object.
LiteMat lite_mat_cut;

// The image is cropped by the values of x, y, w, h.
Crop(lite_mat_bgr, lite_mat_cut, 16, 16, 224, 224);
```

### Normalize image data

In order to eliminate the dimensional influence among the data indicators, and solve the comparability problem among the data indicators through standardization processing, here is the use of the [SubStractMeanNormalize](https://www.mindspore.cn/lite/docs/en/master/apicc/dataset.html#substractmeannormalize) function in `image_process.h` to normalize the image data.

```
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, float *mean, float *norm);
```

#### Usage example

```
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// The mean value of the image data.
// The variance of the image data.
float means[1] = {0.485};
float norm[1] = {1.0 / 0.229};

// Create a normalized image object.
LiteMat lite_mat_bgr_norm;

// The image data is normalized by the mean value and variance of the image data.
SubStractMeanNormalize(lite_mat_bgr, lite_mat_bgr_norm, means, norm);
```
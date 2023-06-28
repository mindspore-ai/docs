﻿# Implementing an Image Classification Application

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/lite/source_en/quick_start/quick_start.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

It is recommended that you start from the image classification demo on the Android device to understand how to build the MindSpore Lite application project, configure dependencies, and use related APIs.
     
This tutorial demonstrates the on-device deployment process based on the image classification sample program on the Android device provided by the MindSpore team.  

1. Select an image classification model.
2. Convert the model into a MindSpore Lite model.
3. Use the MindSpore Lite inference model on the device. The following describes how to use the MindSpore Lite C++ APIs (Android JNIs) and MindSpore Lite image classification models to perform on-device inference, classify the content captured by a device camera, and display the most possible classification result on the application's image preview screen.
   
> Click to find [Android image classification models](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite) and [sample code](https://gitee.com/mindspore/mindspore/tree/r1.0/model_zoo/official/lite/image_classification).

## Selecting a Model

The MindSpore team provides a series of preset device models that you can use in your application.  
Click [here](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms) to download image classification models in MindSpore ModelZoo.
In addition, you can use the preset model to perform migration learning to implement your image classification tasks. 

## Converting a Model

After you retrain a model provided by MindSpore, export the model in the [.mindir format](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_model.html#export-mindir-model). Use the MindSpore Lite [model conversion tool](https://www.mindspore.cn/tutorial/lite/en/r1.0/use/converter_tool.html) to convert the .mindir model to a .ms model.

Take the mobilenetv2 model as an example. Execute the following script to convert a model into a MindSpore Lite model for on-device inference.
```bash
./converter_lite --fmk=MINDIR --modelFile=mobilenetv2.mindir --outputFile=mobilenetv2.ms
```

## Deploying an Application

The following section describes how to build and execute an on-device image classification task on MindSpore Lite.

### Running Dependencies

- Android Studio 3.2 or later (Android 4.0 or later is recommended.)
- Native development kit (NDK) 21.3
- [CMake](https://cmake.org/download) 3.10.2  
- Android software development kit (SDK) 26 or later
- JDK 1.8 or later

### Building and Running

1. Load the sample source code to Android Studio and install the corresponding SDK. (After the SDK version is specified, Android Studio automatically installs the SDK.) 

    ![start_home](../images/lite_quick_start_home.png)

    Start Android Studio, click `File > Settings > System Settings > Android SDK`, and select the corresponding SDK. As shown in the following figure, select an SDK and click `OK`. Android Studio automatically installs the SDK.

    ![start_sdk](../images/lite_quick_start_sdk.png)

    (Optional) If an NDK version issue occurs during the installation, manually download the corresponding [NDK version](https://developer.android.com/ndk/downloads) (the version used in the sample code is 21.3). Specify the NDK location in `Android NDK location` of `Project Structure`.

    ![project_structure](../images/lite_quick_start_project_structure.png)

2. Connect to an Android device and runs the image classification application.

    Connect to the Android device through a USB cable for debugging. Click `Run 'app'` to run the sample project on your device.

    ![run_app](../images/lite_quick_start_run_app.PNG)

    For details about how to connect the Android Studio to a device for debugging, see <https://developer.android.com/studio/run/device>.

    The mobile phone needs to turn on "USB debugging mode" for Android Studio to recognize the phone. In general, Huawei mobile phones turn on "USB debugging mode" in Settings -> System and Update -> Developer Options -> USB Debugging.

3. Continue the installation on the Android device. After the installation is complete, you can view the content captured by a camera and the inference result.

    ![result](../images/lite_quick_start_app_result.png)


## Detailed Description of the Sample Program  

This image classification sample program on the Android device includes a Java layer and a JNI layer. At the Java layer, the Android Camera 2 API is used to enable a camera to obtain image frames and process images. At the JNI layer, the model inference process is completed in [Runtime](https://www.mindspore.cn/tutorial/lite/en/r1.0/use/runtime.html).

> This following describes the JNI layer implementation of the sample program. At the Java layer, the Android Camera 2 API is used to enable a device camera and process image frames. Readers are expected to have the basic Android development knowledge.

### Sample Program Structure

```
app
│
├── src/main
│   ├── assets # resource files
|   |   └── mobilenetv2.ms # model file
│   |
│   ├── cpp # main logic encapsulation classes for model loading and prediction
|   |   |── ...
|   |   ├── mindspore_lite_1.0.0-minddata-arm64-cpu` #MindSpore Lite version
|   |   ├── MindSporeNetnative.cpp # JNI methods related to MindSpore calling
│   |   └── MindSporeNetnative.h # header file
│   |
│   ├── java # application code at the Java layer
│   │   └── com.mindspore.himindsporedemo 
│   │       ├── gallery.classify # implementation related to image processing and MindSpore JNI calling
│   │       │   └── ...
│   │       └── widget # implementation related to camera enabling and drawing
│   │           └── ...
│   │   
│   ├── res # resource files related to Android
│   └── AndroidManifest.xml # Android configuration file
│
├── CMakeList.txt # CMake compilation entry file
│
├── build.gradle # Other Android configuration file
├── download.gradle # MindSpore version download
└── ...
```

### Configuring MindSpore Lite Dependencies

When MindSpore C++ APIs are called at the Android JNI layer, related library files are required. You can use MindSpore Lite [source code compilation](https://www.mindspore.cn/tutorial/lite/en/r1.0/use/build.html) to generate the MindSpore Lite version. In this case, you need to use the compile command of generate with image preprocessing module.

In this example, the build process automatically downloads the `mindspore-lite-1.0.0-minddata-arm64-cpu` by the `app/download.gradle` file and saves in the `app/src/main/cpp` directory.

Note: if the automatic download fails, please manually download the relevant library files and put them in the corresponding location.

mindspore-lite-1.0.0-minddata-arm64-cpu.tar.gz [Download link](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.0/lite/android_aarch64/mindspore-lite-1.0.0-minddata-arm64-cpu.tar.gz)

```
android{
    defaultConfig{
        externalNativeBuild{
            cmake{
                arguments "-DANDROID_STL=c++_shared"
            }
        }

        ndk{ 
            abiFilters'armeabi-v7a', 'arm64-v8a'  
        }
    }
}
```

Create a link to the `.so` library file in the `app/CMakeLists.txt` file:

```
# ============== Set MindSpore Dependencies. =============
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/third_party/flatbuffers/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION})
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/ir/dtype)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/include/schema)

add_library(mindspore-lite SHARED IMPORTED )
add_library(minddata-lite SHARED IMPORTED )

set_target_properties(mindspore-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libmindspore-lite.so)
set_target_properties(minddata-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/src/main/cpp/${MINDSPORELITE_VERSION}/lib/libminddata-lite.so)
# --------------- MindSpore Lite set End. --------------------

# Link target library.       
target_link_libraries(
    ...
     # --- mindspore ---
        minddata-lite
        mindspore-lite
    ...
)
```

### Downloading and Deploying a Model File

In this example, the build process automatically downloads the `mobilenetv2.ms` by the `app/download.gradle` file and saves in the `app/src/main/assets` directory.

Note: if the automatic download fails, please manually download the relevant library files and put them in the corresponding location.

mobilenetv2.ms [mobilenetv2.ms]( https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms)

### Compiling On-Device Inference Code

Call MindSpore Lite C++ APIs at the JNI layer to implement on-device inference.

The inference code process is as follows. For details about the complete code, see `src/cpp/MindSporeNetnative.cpp`. 

1. Load the MindSpore Lite model file and build the context, session, and computational graph for inference.  

    - Load a model file. Create and configure the context for model inference.
        ```cpp
        // Buffer is the model data passed in by the Java layer
        jlong bufferLen = env->GetDirectBufferCapacity(buffer);
        char *modelBuffer = CreateLocalModelBuffer(env, buffer);  
        ```
        
    - Create a session.
        ```cpp
        void **labelEnv = new void *;
        MSNetWork *labelNet = new MSNetWork;
        *labelEnv = labelNet;
        
        // Create context.
        mindspore::lite::Context *context = new mindspore::lite::Context;
        context->thread_num_ = num_thread;
        
        // Create the mindspore session.
        labelNet->CreateSessionMS(modelBuffer, bufferLen, "device label", context);
        delete(context);
        
        ```
        
    - Load the model file and build a computational graph for inference.
        ```cpp
        void MSNetWork::CreateSessionMS(char* modelBuffer, size_t bufferLen, std::string name, mindspore::lite::Context* ctx)
        {
            CreateSession(modelBuffer, bufferLen, ctx);  
            session = mindspore::session::LiteSession::CreateSession(ctx);
            auto model = mindspore::lite::Model::Import(modelBuffer, bufferLen);
            int ret = session->CompileGraph(model);
        }
        ```
    
2. Convert the input image into the Tensor format of the MindSpore model. 

    Convert the image data to be detected into the Tensor format of the MindSpore model.

    ```cpp
    // Convert the Bitmap image passed in from the JAVA layer to Mat for OpenCV processing
     BitmapToMat(env, srcBitmap, matImageSrc);
   // Processing such as zooming the picture size.
    matImgPreprocessed = PreProcessImageData(matImageSrc);  

    ImgDims inputDims; 
    inputDims.channel = matImgPreprocessed.channels();
    inputDims.width = matImgPreprocessed.cols;
    inputDims.height = matImgPreprocessed.rows;
    float *dataHWC = new float[inputDims.channel * inputDims.width * inputDims.height]

    // Copy the image data to be detected to the dataHWC array.
    // The dataHWC[image_size] array here is the intermediate variable of the input MindSpore model tensor.
    float *ptrTmp = reinterpret_cast<float *>(matImgPreprocessed.data);
    for(int i = 0; i < inputDims.channel * inputDims.width * inputDims.height; i++){
       dataHWC[i] = ptrTmp[i];
    }

    // Assign dataHWC[image_size] to the input tensor variable.
    auto msInputs = mSession->GetInputs();
    auto inTensor = msInputs.front();
    memcpy(inTensor->MutableData(), dataHWC,
        inputDims.channel * inputDims.width * inputDims.height * sizeof(float));
    delete[] (dataHWC);
   ```
   
3. Preprocessing the input data.

   ```cpp
   bool PreProcessImageData(const LiteMat &lite_mat_bgr, LiteMat *lite_norm_mat_ptr) {
     bool ret = false;
     LiteMat lite_mat_resize;
     LiteMat &lite_norm_mat_cut = *lite_norm_mat_ptr;
     ret = ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
     if (!ret) {
       MS_PRINT("ResizeBilinear error");
       return false;
     }
     LiteMat lite_mat_convert_float;
     ret = ConvertTo(lite_mat_resize, lite_mat_convert_float, 1.0 / 255.0);
     if (!ret) {
       MS_PRINT("ConvertTo error");
       return false;
     }
     LiteMat lite_mat_cut;
     ret = Crop(lite_mat_convert_float, lite_mat_cut, 16, 16, 224, 224);
     if (!ret) {
       MS_PRINT("Crop error");
       return false;
     }
     float means[3] = {0.485, 0.456, 0.406};
     float vars[3] = {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225};
     SubStractMeanNormalize(lite_mat_cut, lite_norm_mat_cut, means, vars);
     return true;
   }
   ```

4. Perform inference on the input tensor based on the model, obtain the output tensor, and perform post-processing.    

   - Perform graph execution and on-device inference.

        ```cpp
        // After the model and image tensor data is loaded, run inference.
        auto status = mSession->RunGraph();
        ```

   - Obtain the output data.
        ```cpp
        auto names = mSession->GetOutputTensorNames();
        std::unordered_map<std::string,mindspore::tensor::MSTensor *> msOutputs;
        for (const auto &name : names) {
            auto temp_dat =mSession->GetOutputByTensorName(name);
            msOutputs.insert(std::pair<std::string, mindspore::tensor::MSTensor *> {name, temp_dat});
          }
        std::string retStr = ProcessRunnetResult(msOutputs, ret);
        ```
        
   - Perform post-processing of the output data.
        ```cpp
        std::string ProcessRunnetResult(std::unordered_map<std::string,
                mindspore::tensor::MSTensor *> msOutputs, int runnetRet) {
        
          std::unordered_map<std::string, mindspore::tensor::MSTensor *>::iterator iter;
          iter = msOutputs.begin();
        
          // The mobilenetv2.ms model output just one branch.
          auto outputTensor = iter->second;
          int tensorNum = outputTensor->ElementsNum();
        
          // Get a pointer to the first score.
          float *temp_scores = static_cast<float * >(outputTensor->MutableData());
        
          float scores[RET_CATEGORY_SUM];
          for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
            if (temp_scores[i] > 0.5) {
              MS_PRINT("MindSpore scores[%d] : [%f]", i, temp_scores[i]);
            }
            scores[i] = temp_scores[i];
          }
        
          // Score for each category.
          // Converted to text information that needs to be displayed in the APP.
          std::string categoryScore = "";
          for (int i = 0; i < RET_CATEGORY_SUM; ++i) {
            categoryScore += labels_name_map[i];
            categoryScore += ":";
            std::string score_str = std::to_string(scores[i]);
            categoryScore += score_str;
            categoryScore += ";";
          }
          return categoryScore;
        }      
        ```
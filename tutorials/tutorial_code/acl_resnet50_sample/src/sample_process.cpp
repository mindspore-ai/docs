/**
* @file sample_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "inc/sample_process.h"
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "inc/model_process.h"
#include "acl/acl.h"
#include "inc/utils.h"
extern bool g_isDevice;

SampleProcess::SampleProcess() :deviceId_(0), context_(nullptr), stream_(nullptr) {
}

SampleProcess::~SampleProcess() {
    DestroyResource();
}

Result SampleProcess::InitResource() {
    // ACL init
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("get run mode success");
    return SUCCESS;
}

void SampleProcess::GetAllFiles(std::string path, std::vector<string> *files) {
    DIR *pDir;
    struct dirent* ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            files->push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}

Result SampleProcess::Process(char *om_path, char *input_folder) {
    // model init
    ModelProcess processModel;
    // const char* omModelPath = "../model/resnet50.om";
    const char* omModelPath = om_path;

    Result ret = processModel.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = processModel.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = processModel.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    std::vector<string> testFile;
    GetAllFiles(input_folder, &testFile);

    if (testFile.size() == 0) {
        WARN_LOG("no input data under folder");
    }

    // loop begin
    for (size_t index = 0; index < testFile.size(); ++index) {
        INFO_LOG("start to process file:%s", testFile[index].c_str());
        // model process
        uint32_t devBufferSize;
        void *picDevBuffer = Utils::GetDeviceBufferOfFile(testFile[index], &devBufferSize);
        if (picDevBuffer == nullptr) {
            ERROR_LOG("get pic device buffer failed,index is %zu", index);
            return FAILED;
        }
        ret = processModel.CreateInput(picDevBuffer, devBufferSize);
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }

        ret = processModel.Execute();
        if (ret != SUCCESS) {
            ERROR_LOG("execute inference failed");
            aclrtFree(picDevBuffer);
            return FAILED;
        }

        int pos = testFile[index].find_last_of('/');
        std::string name = testFile[index].substr(pos+1);
        std::string outputname = name.substr(0, name.rfind("."));

        // print the top 5 confidence values
        processModel.OutputModelResult();
        // dump output result to file in the current directory
        processModel.DumpModelOutputResult(reinterpret_cast<char *>(outputname.c_str()));

        // release model input buffer
        aclrtFree(picDevBuffer);
        processModel.DestroyInput();
    }
    // loop end

    return SUCCESS;
}

void SampleProcess::DestroyResource() {
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}


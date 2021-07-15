/**
* @file sample_process.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <string>
#include <vector>
#include "inc/utils.h"
#include "acl/acl.h"

/**
* SampleProcess
*/
class SampleProcess {
 public:
    /**
    * @brief Constructor
    */
    SampleProcess();

    /**
    * @brief Destructor
    */
    ~SampleProcess();

    /**
    * @brief init reousce
    * @return result
    */
    Result InitResource();

    /**
    * @brief sample process
    * @return result
    */
    Result Process(char *om_path, char *input_folder);

    void GetAllFiles(std::string path, std::vector<std::string> *files);

 private:
    void DestroyResource();

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
};


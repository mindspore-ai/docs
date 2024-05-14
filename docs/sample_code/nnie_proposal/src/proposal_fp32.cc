/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/proposal_fp32.h"
#include <memory>
#include <string>
#include "include/schema/model_generated.h"
#include "include/registry/register_kernel.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;
#define MAX_SIZE 1024

namespace mindspore {
namespace proposal {
int ProposalCPUKernel::Prepare() {
  if (inputs_.size() < 2) {
    LOGE("inputs tensor num error.");
    return RET_ERROR;
  }
  if (outputs_.size() != 1) {
    LOGE("outputs tensor num error.");
    return RET_ERROR;
  }
  std::vector<std::string> inputs_name = {"rpn_cls_score", "rpn_bbox_pred"};
  std::vector<mindspore::tensor::MSTensor *> inputs;
  for (size_t i = 0; i < inputs_name.size(); i++) {
    bool find_flag = false;
    for (auto &input : inputs_) {
      if (input->tensor_name() == inputs_name[i]) {
        inputs.push_back(input);
        find_flag = true;
        break;
      }
    }
    if (!find_flag) {
      for (auto &input : inputs_) {
        if (std::find(inputs.begin(), inputs.end(), input) != inputs.end()) {
          continue;
        }
        inputs.push_back(input);
        LOGW("input tensor name diff '%s' vs '%s'.", inputs_name[i].c_str(), input->tensor_name().c_str());
        break;
      }
    }
  }
  if (inputs.size() != inputs_name.size()) {
    LOGE("inputs size error.");
    return RET_ERROR;
  }

  this->set_inputs(inputs);
  if (inputs[0]->shape()[0] != 1) {
    LOGE("proposal only support input num == 1.");
    return RET_ERROR;
  }

  outputs_[0]->set_tensor_name("proposal");

  int max_roi_num_int = 300;
  auto *max_roi_num = std::getenv("MAX_ROI_NUM");
  if (max_roi_num != nullptr) {
    auto iter =
      std::find_if(max_roi_num, max_roi_num + strlen(max_roi_num), [](char val) { return val < '0' || val > '9'; });
    if (iter != max_roi_num) {
      *iter = '\0';
      max_roi_num_int = atoi(max_roi_num);
    } else {
      LOGW("MAX_ROI_NUM ENV is invalid, now set to default value %d", max_roi_num_int);
    }
  } else {
    LOGW("MAX_ROI_NUM ENV is not set, now set to default value %d", max_roi_num_int);
  }

  return ProposalInit(&proposal_param_, &inputs_, max_roi_num_int, image_height_, image_weight_);
}

int ProposalCPUKernel::ReSize() {
  if (inputs_[0]->shape()[0] != 1) {
    LOGE("proposal only support input num == 1.");
    return RET_ERROR;
  }
  return RET_OK;
}

int ProposalCPUKernel::Execute() { return ProposalRun(&inputs_, &outputs_, &proposal_param_); }

ProposalCPUKernel::~ProposalCPUKernel() { ProposalDeInit(&proposal_param_); }

bool GetCustomAttr(char *buf, int buf_size, const mindspore::schema::Custom *op, const std::string &attr) {
  int attr_size;
  for (size_t i = 0; i < op->attr()->size(); i++) {
    if (op->attr()->Get(i)->name()->str() == attr) {
      auto output_info = op->attr()->Get(i)->data();
      attr_size = static_cast<int>(output_info->size());
      if (attr_size >= buf_size) {
        LOGE("attr size too big");
        return false;
      }
      for (int j = 0; j < attr_size; j++) {
        buf[j] = static_cast<char>(output_info->Get(j));
      }
      buf[attr_size] = 0;
      return true;
    }
  }
  return false;
}

std::shared_ptr<mindspore::kernel::Kernel> ProposalCreateKernel(
  const std::vector<mindspore::tensor::MSTensor *> &inputs, const std::vector<mindspore::tensor::MSTensor *> &outputs,
  const mindspore::schema::Primitive *primitive, const mindspore::lite::Context *ctx) {
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    LOGE("Primitive type is not PrimitiveType_Custom");
    return nullptr;
  }

  auto op = primitive->value_as_Custom();
  if (op->attr()->size() < 1) {
    LOGE("There are at least 1 attribute of Custom");
    return nullptr;
  }
  int64_t ndims;
  int64_t image_height;
  int64_t image_width;

  char *res = nullptr;
  char buf[MAX_SIZE];
  if (GetCustomAttr(buf, MAX_SIZE, op, "proposal_id")) {
    res = nullptr;
    ndims = strtol(buf, &res, 10);
    if ((*res) != 0) {
      LOGE("Get attr id data fail");
      return nullptr;
    }
  } else {
    LOGE("Proposal Custom op should have id");
    return nullptr;
  }

  if (GetCustomAttr(buf, MAX_SIZE, op, "image_height")) {
    res = nullptr;
    image_height = strtol(buf, &res, 10);
    if ((*res) != 0) {
      LOGE("Get attr id data fail");
      return nullptr;
    }
  } else {
    LOGE("Proposal Custom op should have image_height");
    return nullptr;
  }
  if (GetCustomAttr(buf, MAX_SIZE, op, "image_width")) {
    res = nullptr;
    image_width = strtol(buf, &res, 10);
    if ((*res) != 0) {
      LOGE("Get attr id data fail");
      return nullptr;
    }
  } else {
    LOGE("Proposal Custom op should have image_width");
    return nullptr;
  }

  auto kernel = std::make_shared<ProposalCPUKernel>(inputs, outputs, primitive, ctx, ndims, image_height, image_width);
  // auto kernel = new (std::nothrow) ProposalCPUKernel(inputs, outputs, primitive, ctx, ndims, image_height,
  // image_width);
  if (kernel == nullptr) {
    LOGE("new custom kernel is nullptr");
    return nullptr;
  }
  return kernel;
}
}  // namespace proposal
}  // namespace mindspore

namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL(CPU, NNIE, kNumberTypeFloat32, Proposal, proposal::ProposalCreateKernel)
}  // namespace kernel
}  // namespace mindspore

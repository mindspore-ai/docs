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

#include "src/proposal_infer.h"
#include <memory>
#include <vector>
#include "include/errorcode.h"
#include "src/proposal.h"

using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace proposal {
std::shared_ptr<KernelInterface> ProposalInferCreater() {
  auto infer = std::make_shared<ProposalInterface>();
  if (infer == nullptr) {
    LOGE("new custom infer is nullptr");
    return nullptr;
  }

  return infer;
}
int ProposalInterface::Infer(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                             const std::vector<mindspore::tensor::MSTensor *> &outputs,
                             const mindspore::schema::Primitive *primitive) {
  if (inputs.size() != 2) {
    LOGE("Inputs size less 2");
    return RET_ERROR;
  }
  if (outputs.size() == 0) {
    LOGE("Outputs size 0");
    return RET_ERROR;
  }
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    LOGE("Primitive type is not PrimitiveType_Custom");
    return RET_ERROR;
  }

  size_t id = 0;
  while (id < outputs.size()) {
    // 待补完
    // outputs[id]->format_ = input->format_;
    // outputs[id]->data_type_ = kNumberTypeFloat32;
    // 设置type为int
    std::vector<int> shape{-1, COORDI_NUM};
    outputs[id]->set_shape(shape);
    id++;
  }
  return RET_OK;
}
}  // namespace proposal
}  // namespace mindspore
namespace mindspore {
namespace kernel {
// static KernelInterfaceReg a(aa, schema::PrimitiveType_Custom, CustomInferCreater);
REGISTER_CUSTOM_KERNEL_INTERFACE(NNIE, Proposal, proposal::ProposalInferCreater);
}  // namespace kernel
}  // namespace mindspore

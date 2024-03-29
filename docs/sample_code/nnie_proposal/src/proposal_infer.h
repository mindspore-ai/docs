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

#ifndef MINDSPORE_LITE_TOOLS_BENCHMARK_NNIE_PROPOSAL_PROPOSAL_INFER_H_
#define MINDSPORE_LITE_TOOLS_BENCHMARK_NNIE_PROPOSAL_PROPOSAL_INFER_H_
#include <vector>
#include "include/registry/kernel_interface.h"

namespace mindspore {
namespace proposal {
class ProposalInterface : public mindspore::kernel::KernelInterface {
 public:
  ProposalInterface() {}

  ~ProposalInterface() = default;

  int Infer(const std::vector<mindspore::tensor::MSTensor *> &inputs,
            const std::vector<mindspore::tensor::MSTensor *> &outputs,
            const mindspore::schema::Primitive *primitive) override;
};
}  // namespace proposal
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_BENCHMARK_NNIE_PROPOSAL_PROPOSAL_INFER_H_

// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/kernels/selected_rows/clip_by_norm_kernel.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void ClipByNormSparseFunctor(const Context& dev_ctx,
                             const SelectedRows& in,
                             float max_norm,
                             SelectedRows* output) {
  phi::SelectedRows merged_input;
  // dev_ctx.template Alloc<T>(&merger_input);
  paddle::operators::math::scatter::MergeAdd<Context, T> merge_func;
  merge_func(dev_ctx, in, &merged_input);
  auto input = &(merged_input.value());
  output->set_rows(merged_input.rows());
  output->set_height(merged_input.height());
  auto out_tensor = output->mutable_value();
  out_tensor->Resize(merged_input.value().dims());
  dev_ctx.template Alloc<T>(out_tensor);

  PADDLE_ENFORCE_NOT_NULL(input,
                          phi::errors::InvalidArgument(
                              "Input(X) of ClipByNormOp should not be null. "
                              "Please check if it is created correctly."));

  auto x = paddle::framework::EigenVector<T>::Flatten(*input);
  auto out = paddle::framework::EigenVector<T>::Flatten(*out_tensor);
  auto x_norm = x.square().sum().sqrt();
  auto* place = dev_ctx.eigen_device();

  auto temp = (x_norm <= max_norm).template cast<T>();
  auto epsilon = ((x_norm <= static_cast<T>(1e-30)).all().template cast<T>()) *
                 static_cast<T>(1e-6);

  auto scaling =
      temp + (static_cast<T>(1) - temp) * max_norm / (x_norm + epsilon);
  Eigen::array<int, 1> one_dim{{1}};
  Eigen::DSizes<int, 1> m_dsize(input->numel());
  if (dev_ctx.GetPlace() == phi::CPUPlace()) {
    out.device(*place) = x * scaling.reshape(one_dim).eval().broadcast(m_dsize);
  } else {
    out.device(*place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
  }
}

}  // namespace sr
}  // namespace phi

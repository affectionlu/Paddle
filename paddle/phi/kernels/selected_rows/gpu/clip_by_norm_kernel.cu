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

#include "paddle/phi/kernels/selected_rows/clip_by_norm_kernel.h"

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/selected_rows/impl/clip_by_norm_kernel_impl.h"

namespace phi {
namespace sr {

template <>
void ClipByNormSparseKernel<dtype::float16,
                            paddle::platform::CUDADeviceContext>(
    const paddle::platform::CUDADeviceContext& dev_ctx,
    const SelectedRows& in,
    float max_norm,
    SelectedRows* output) {
  phi::SelectedRows merged_input;
  paddle::operators::math::scatter::MergeAdd<Context, dtype::float16>
      merge_func;
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
  std::vector<int> reduce_dims;
  reduce_dims.resize(input->dims().size());
  for (int i = 0; i < reduce_dims.size(); ++i) {
    reduce_dims[i] = i;
  }
  DenseTensor* tmp;
  tmp->Resize({1});
  dev_ctx.template Alloc<float>(tmp);
  phi::funcs::ReduceKernel<dtype::float16,
                           float,
                           kps::AddFunctor,
                           kps::SquareFunctor<dtype::float16, float>>(
      dev_ctx,
      input->value(),
      tmp,
      kps::SquareFunctor<dtype::float16, float>(),
      reduce_dims);
  auto tmp_eigen = paddle::framework::EigenVector<float>::Flatten(*tmp);
  auto x_norm = tmp_eigen.sqrt();

  auto x =
      paddle::framework::EigenVector<dtype::float16>::Flatten(input->value());
  auto out =
      paddle::framework::EigenVector<dtype::float16>::Flatten(*out_tensor);
  auto* place = dev_ctx.eigen_device();

  auto temp = (x_norm <= max_norm).template cast<float>();
  auto epsilon =
      ((x_norm <= static_cast<float>(1e-30)).all().template cast<float>()) *
      static_cast<float>(1e-6);

  auto scaling =
      (temp + (static_cast<float>(1) - temp) * max_norm / (x_norm + epsilon))
          .template cast<dtype::float16>();
  Eigen::array<int, 1> one_dim{{1}};
  Eigen::DSizes<int, 1> m_dsize(input->numel());

  out.device(*place) = x * scaling.reshape(one_dim).broadcast(m_dsize);
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(clip_by_norm_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::ClipByNormSparseKernel,
                   float,
                   phi::dtype::float16) {}

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class FillConstantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));

    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");
    auto force_cpu = ctx.Attr<bool>("force_cpu");
    auto place_type = ctx.Attr<int>("place_type");
    framework::Tensor *tensor = nullptr;

    framework::Variable *out_var = ctx.OutputVar("Out");

    T value;
    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream.
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<double>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<double>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        if (std::is_same<int64_t, T>::value) {
          int64_t tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        } else {
          double tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        }
      }
    }
    if (ctx.HasInput("ValueTensor")) {
      auto *value_tensor = ctx.Input<framework::Tensor>("ValueTensor");
      PADDLE_ENFORCE_EQ(
          value_tensor->numel(),
          1,
          platform::errors::InvalidArgument(
              "When use Tensor as value to set Tensor value in fill_cosntant, "
              "value input(ValueTensor) size must be 1, but get %d",
              value_tensor->numel()));
      const T *tensor_data = value_tensor->data<T>();
      framework::Tensor cpu_tensor;
      auto tmp_place = value_tensor->place();
      if (platform::is_gpu_place(tmp_place) ||
          platform::is_xpu_place(tmp_place)) {
        paddle::framework::TensorCopySync(
            *value_tensor, platform::CPUPlace(), &cpu_tensor);
        tensor_data = cpu_tensor.data<T>();
      }
      value = tensor_data[0];
    }
    auto shape = GetShape(ctx);

    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      tensor->Resize(shape);
    } else if (out_var->IsType<phi::SelectedRows>()) {
      tensor = out_var->GetMutable<phi::SelectedRows>()->mutable_value();
      tensor->Resize(shape);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "In fill constant Op, the output only supports SelectedRows and "
          "LoDTensor."));
    }

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    int actual_place = place_type;

    if (actual_place == -1) {
      bool cpu_place = (force_cpu || ctx.GetPlace() == platform::CPUPlace() ||
                        data_type == framework::proto::VarType::BF16);
      if (cpu_place) {
        actual_place = 0;
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        actual_place = 1;
      } else if (platform::is_xpu_place(ctx.GetPlace())) {
        actual_place = 3;
      }
    }

    if (actual_place == 0) {
      VLOG(4) << "[CPU] FillConstantKernel"
              << ((data_type == framework::proto::VarType::BF16) ? "<bfloat16>"
                                                                 : "<T>");
      tensor->mutable_data(platform::CPUPlace(),
                           framework::TransToPhiDataType(data_type));
      phi::funcs::SetConstant<phi::CPUContext, T> functor;
      auto &dev_ctx = *pool.Get(platform::CPUPlace());
      functor(reinterpret_cast<const phi::CPUContext &>(dev_ctx),
              tensor,
              static_cast<T>(value));
    } else if (actual_place == 1) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      tensor->mutable_data(ctx.GetPlace(),
                           framework::TransToPhiDataType(data_type));
      phi::funcs::SetConstant<platform::CUDADeviceContext, T> functor;
      auto &dev_ctx = *pool.Get(ctx.GetPlace());
      functor(reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx),
              tensor,
              static_cast<T>(value));
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU."));
#endif
    } else if (actual_place == 2) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      tensor->mutable_data(platform::CUDAPinnedPlace(),
                           framework::TransToPhiDataType(data_type));
      phi::funcs::SetConstant<platform::CUDAPinnedDeviceContext, T> functor;
      auto &dev_ctx = *pool.Get(platform::CUDAPinnedPlace());
      functor(
          reinterpret_cast<const platform::CUDAPinnedDeviceContext &>(dev_ctx),
          tensor,
          static_cast<T>(value));
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU."));
#endif
    } else if (actual_place == 3) {
#ifdef PADDLE_WITH_XPU
      tensor->mutable_data(ctx.GetPlace(),
                           framework::TransToPhiDataType(data_type));
      phi::funcs::SetConstant<platform::XPUDeviceContext, T> functor;
      auto &dev_ctx = *pool.Get(ctx.GetPlace());
      functor(reinterpret_cast<const platform::XPUDeviceContext &>(dev_ctx),
              tensor,
              static_cast<T>(value));
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU."));
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Could NOT determine the place of variable, place_type = %d .",
          actual_place));
    }
  }
};
}  // namespace operators
}  // namespace paddle

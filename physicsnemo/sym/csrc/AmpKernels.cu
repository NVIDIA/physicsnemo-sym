/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
// Modified from https://github.com/pytorch/pytorch/blob/release/1.11/aten/src/ATen/native/cuda/AmpKernels.cu#L181

#include <torch/python.h>
#include <c10/cuda/CUDAStream.h>

using torch::Tensor;

// amp_update_scale_cuda_kernel is launched with a single thread to compute the new scale.
// The scale factor is maintained and updated on the GPU to avoid synchronization.
__global__ void amp_update_scale_cuda_kernel(float* current_scale,
                                             int* growth_tracker,
                                             float* found_inf,
                                             float growth_factor,
                                             float backoff_factor,
                                             int growth_interval,
                                             float max_scale,
                                             float recover_threshold,
                                             int recover_growth_interval) {
  if (*found_inf) {
    *current_scale = (*current_scale)*backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    // decide whether to use the recover_growth_interval
    growth_interval = (*current_scale) <= recover_threshold ? recover_growth_interval : growth_interval;
    if (successful == growth_interval) {
      // grow the scale then clamp with max_scale
      *current_scale = min((*current_scale) * growth_factor, max_scale);
      *growth_tracker = 0;
    } else {
      *growth_tracker = successful;
    }
  }
}

// _amp_update_scale_cuda asynchronously updates the scale tensor in place.
//
// Args:
// current_scale:  A one-element cuda float tensor containing the scale value.
// growth_tracker:  A one-element torch.cuda.IntTensor containing the number of recent consecutive unskipped steps.
// found_inf:  A one-element cuda float tensor. If > 0, indicates that infs/nans were found by the relevant
//             prior _amp_non_finite_check_and_unscale_cuda call, and 0 if no infs/nans were found.
// growth_factor:  Multiplier if no infs/NaNs were found (typically slightly > 1).
// backoff_factor:  Multiplier if infs/NaNs were found (typically 0.5).
// growth_interval:  Number of consecutive unskipped steps that must occur for current_scale to be multiplied by
//                   growth_factor.
// max_scale:  The maximum value the scale could grow.
// recover_threshold:  Allowing quickly recover the scaling factor when it is less or equal than this threshold.
// recover_growth_interval:  The growth_interval that will be used when the scaling factor is less or equal than
//                           the recover_threshold.
//
// Returns:
// current_scale
Tensor& _amp_update_scale_cuda_(Tensor& current_scale,
                                Tensor& growth_tracker,
                                const Tensor& found_inf,
                                double growth_factor,
                                double backoff_factor,
                                int64_t growth_interval,
                                double max_scale,
                                double recover_threshold,
                                int64_t recover_growth_interval)
{
  TORCH_CHECK(growth_tracker.is_cuda(), "growth_tracker must be a CUDA tensor.");
  TORCH_CHECK(current_scale.is_cuda(), "current_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(growth_tracker.numel() == 1, "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int, "growth_tracker must be an int tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  amp_update_scale_cuda_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
    current_scale.data_ptr<float>(),
    growth_tracker.data_ptr<int>(),
    found_inf.data_ptr<float>(),
    growth_factor,
    backoff_factor,
    growth_interval,
    max_scale,
    recover_threshold,
    recover_growth_interval);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return current_scale;
}

TORCH_LIBRARY(physicsnemo_ext, m) {
  m.def("_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval, float max_scale, float recover_threshold, int recover_growth_interval) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(physicsnemo_ext, CUDA, m) {
  m.impl("_amp_update_scale_", _amp_update_scale_cuda_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

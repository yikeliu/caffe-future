// Copyright 2014 kloudkl@github

#include <cmath>  // for std::abs

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/regularizer.hpp"
#include "caffe/util/math_functions.hpp"  // for caffe_gpu_asum

namespace caffe {

template <typename Dtype>
__device__ inline int gpu_sign(const Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

template __device__ int gpu_sign<float>(const float val);
template __device__ int gpu_sign<double>(const double val);

template <typename Dtype>
__global__ void ScaleSign(const int n, const Dtype coeff, const Dtype* data,
                          Dtype* diff) {
  CUDA_KERNEL_LOOP(index, n) {
    diff[index] += coeff * gpu_sign<Dtype>(data[index]);
  }
}

template <typename Dtype>
Dtype L1Regularizer<Dtype>::Loss_gpu(Blob<Dtype>* top) {
  const Dtype* data = top->gpu_data();
  Dtype penalty;
  caffe_gpu_asum<Dtype>(top->count(), data, &penalty);
  return this->coeff_ * penalty;
}

template <typename Dtype>
void L1Regularizer<Dtype>::Gradient_gpu(Blob<Dtype>* top) {
  const Dtype* data = top->gpu_data();
  Dtype* diff = top->mutable_gpu_diff();
  int count = top->count();
  /* NOLINT_NEXT_LINE(whitespace/operators) */
  ScaleSign<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, this->coeff_, data, diff);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
Dtype L2Regularizer<Dtype>::Loss_gpu(Blob<Dtype>* top) {
  const Dtype* data = top->gpu_data();
  Dtype penalty;
  caffe_gpu_dot<Dtype>(top->count(), data, data, &penalty);
  return this->coeff_ * penalty;
}

template <typename Dtype>
void L2Regularizer<Dtype>::Gradient_gpu(Blob<Dtype>* top) {
  const Dtype* data = top->gpu_data();
  Dtype* diff = top->mutable_gpu_diff();
  caffe_gpu_axpy<Dtype>(top->count(), this->coeff_ * 2., data, diff);
}

template <typename Dtype>
Dtype MaxNormRegularizer<Dtype>::Loss_gpu(Blob<Dtype>* top) {
  // TODO: Implement MaxNormRegularizer::Loss_gpu
  return this->coeff_ * 0;
}

template <typename Dtype>
void MaxNormRegularizer<Dtype>::Gradient_gpu(Blob<Dtype>* top) {
  // TODO: Implement MaxNormRegularizer::Gradient_gpu
  const Dtype* data = top->cpu_data();
  Dtype* diff = top->mutable_cpu_diff();
}

INSTANTIATE_CLASS(Regularizer);
INSTANTIATE_CLASS(L1Regularizer);
INSTANTIATE_CLASS(L2Regularizer);
INSTANTIATE_CLASS(MaxNormRegularizer);

}  // namespace caffe

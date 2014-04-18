// Copyright 2014 kloudkl@github

#include <cmath>  // for std::abs

#include "caffe/proto/caffe.pb.h"
#include "caffe/regularizer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype Regularizer<Dtype>::Loss(Blob<Dtype>* top) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return Loss_cpu(top);
  case Caffe::GPU:
    return Loss_gpu(top);
  default:
    LOG(FATAL) << "Unknown mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void Regularizer<Dtype>::Gradient(Blob<Dtype>* top) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Gradient_cpu(top);
    break;
  case Caffe::GPU:
    Gradient_gpu(top);
    break;
  default:
    LOG(FATAL) << "Unknown mode: " << Caffe::mode();
  }
}

template <typename Dtype>
Dtype L1Regularizer<Dtype>::Loss_cpu(Blob<Dtype>* top) {
  const Dtype* data = top->cpu_data();
  return this->coeff_ * caffe_cpu_asum(top->count(), data);
}

template <typename Dtype>
void L1Regularizer<Dtype>::Gradient_cpu(Blob<Dtype>* top) {
  const Dtype* data = top->cpu_data();
  Dtype* diff = top->mutable_cpu_diff();
  for (int c = 0; c < top->count(); ++c) {
    diff[c] += this->coeff_ * caffe_sign<Dtype>(data[c]);
  }
}

template <typename Dtype>
Dtype L2Regularizer<Dtype>::Loss_cpu(Blob<Dtype>* top) {
  const Dtype* data = top->cpu_data();
  return this->coeff_ * caffe_cpu_dot<Dtype>(top->count(), data, data);
}

template <typename Dtype>
void L2Regularizer<Dtype>::Gradient_cpu(Blob<Dtype>* top) {
  const Dtype* data = top->cpu_data();
  Dtype* diff = top->mutable_cpu_diff();
  caffe_axpy<Dtype>(top->count(), this->coeff_ * 2., data, diff);
}

template <typename Dtype>
Dtype MaxNormRegularizer<Dtype>::Loss_cpu(Blob<Dtype>* top) {
  // TODO: Implement MaxNormRegularizer::Loss_cpu
  return this->coeff_ * 0;
}

template <typename Dtype>
void MaxNormRegularizer<Dtype>::Gradient_cpu(Blob<Dtype>* top) {
  // TODO: Implement MaxNormRegularizer::Gradient_cpu
  const Dtype* data = top->cpu_data();
  Dtype* diff = top->mutable_cpu_diff();
}

template <typename Dtype>
Regularizer<Dtype>* GetRegularizer(const RegularizerParameter& param) {
  const RegularizerParameter_RegularizerType type = param.type();
  switch (type) {
  case REG_TYPE(L1):
    return new L1Regularizer<Dtype>(param);
  case REG_TYPE(L2):
    return new L2Regularizer<Dtype>(param);
  case REG_TYPE(MAX_NORM):
    return new MaxNormRegularizer<Dtype>(param);
  default:
    LOG(FATAL) << "Unknown regularizer type: " << type;
  }
  // just to suppress old compiler warnings.
  return (Regularizer<Dtype>*) (NULL);
}

template Regularizer<float>* GetRegularizer<float>(
    const RegularizerParameter& param);
template Regularizer<double>* GetRegularizer<double>(
    const RegularizerParameter& param);

INSTANTIATE_CLASS(Regularizer);
INSTANTIATE_CLASS(L1Regularizer);
INSTANTIATE_CLASS(L2Regularizer);
INSTANTIATE_CLASS(MaxNormRegularizer);

}  // namespace caffe

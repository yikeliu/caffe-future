// Copyright 2014 kloudkl@github

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
using std::vector;

template<typename Dtype>
Dtype RegularizerAsLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CUDA_CHECK(
    cudaMemset(bottom[0]->mutable_gpu_diff(), 0,
               bottom[0]->count() * sizeof(Dtype)));
  return Dtype(0);
}

INSTANTIATE_CLASS(RegularizerAsLossLayer);

}  // namespace caffe

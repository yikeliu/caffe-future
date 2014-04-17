// Copyright 2014 kloudkl@github

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
using std::vector;

template<typename Dtype>
void RegularizerAsLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                          vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) <<
      "RegularizerAsLossLayer takes one blob as input.";
  CHECK_EQ(top->size(), 0) <<
      "RegularizerAsLossLayer takes no blob as output.";
}

template<typename Dtype>
Dtype RegularizerAsLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  return Dtype(0);
}

INSTANTIATE_CLASS(RegularizerAsLossLayer);

}  // namespace caffe

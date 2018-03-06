#include <functional>
#include <utility>
#include <vector>
#include <cfloat>
#include "caffe/layers/CS_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CSLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape()[0], bottom[1]->shape()[0]);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  L_ = this->layer_param_.cs_param().lll();
}

template <typename Dtype>
void CSLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // CS is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int pndim = bottom[0]->shape()[1]; // prediction channels
  int lndim = 0; // label channels
  if (bottom[1]->num_axes() == 1) {
      lndim = 1;
  } else {
      lndim = bottom[1]->shape()[1];
  }
  Dtype cs = Dtype(0.0);
  for (int n=0; n<bottom[0]->shape()[0]; ++n) {
    Dtype pred_age = 0;
    if (pndim > 1) {
        std::vector<Dtype> v0(pndim, 0);
        for (int i=0; i<pndim; ++i) v0[i] = bottom[0]->data_at(n, i, 0, 0);
        pred_age = std::distance(v0.begin(), std::max_element(v0.begin(), v0.end()));
    } else {
        pred_age = bottom[0]->data_at(n, 0, 0, 0);
    }
    Dtype label_age = 0;
    if (lndim > 1) {
        std::vector<Dtype> v0(lndim, 0);
        for (int i=0; i<lndim; ++i) v0[i] = bottom[1]->data_at(n, i, 0, 0);
        label_age = std::distance(v0.begin(), std::max_element(v0.begin(), v0.end()));
    } else {
        label_age = bottom[1]->data_at(n, 0, 0, 0);
    }
    CHECK_GE(label_age, 0)<<"label-age >= 0";
    CHECK_GE(pred_age, 0)<<"predicted-age >= 0";
    if (std::abs(label_age - pred_age)<=L_) cs+=1;
  }
  cs /= bottom[0]->shape()[0];
  top[0]->mutable_cpu_data()[0] = cs;  
}

INSTANTIATE_CLASS(CSLayer);
REGISTER_LAYER_CLASS(CS);

}  // namespace caffe

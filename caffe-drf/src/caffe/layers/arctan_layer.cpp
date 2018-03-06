#include <functional>
#include <utility>
#include <vector>
#include <cfloat>
#include <cmath>
#include "caffe/layers/arctan_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifndef PI
#define PI 3.1415926
#endif

namespace caffe {
template <typename Dtype>
void ArctanLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //CHECK_EQ(bottom[0]->shape()[0], bottom[1]->shape()[0]);
  CHECK_EQ(bottom[0]->shape()[1], 2);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
}

template <typename Dtype>
void ArctanLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(4);  // MAE is a scalar; 0 axes.
  top_shape[0]= bottom[0]->shape()[0];
  top_shape[1]= 1;
  top_shape[2]= 1;
  top_shape[3]= 1;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ArctanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n=0; n<bottom[0]->shape()[0]; ++n) {
    Dtype ang_temp;
    ang_temp = atan2(bottom[0]->data_at(n,1,0,0), bottom[0]->data_at(n,0,0,0))/PI* (Dtype)180.0 ;

    top_data[top[0]->offset(n,0,0,0)] = ang_temp>=0?ang_temp:(ang_temp+360);
    //LOG(INFO)<<top_data[top[0]->offset(n,0,0,0)]<<"," <<bottom[0]->data_at(n,0,0,0)<<","<<bottom[0]->data_at(n,1,0,0)<<","<<atan2(bottom[0]->data_at(n,1,0,0), bottom[0]->data_at(n,0,0,0))/PI* (Dtype)180.0 ;
}
}

INSTANTIATE_CLASS(ArctanLayer);
REGISTER_LAYER_CLASS(Arctan);

}  // namespace caffe

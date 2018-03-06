#include <functional>
#include <utility>
#include <vector>
#include <cfloat>
#include "caffe/layers/KLD_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void KLDLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void KLDLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void KLDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* pred = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype kld = 0.0;
  int N = bottom[0]->count(0,1);
  int D = bottom[0]->count(1);
  for (int n=0; n<N; ++n) {
    for (int d=0; d<D; ++d) {
      const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
      const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
      
      kld += q * log(q / p);
    }
  }
  kld = kld / N;
  // see http://cse.seu.edu.cn/PersonalPage/xgeng/LDL/resource/tkde16.pdf eq(1)
  top[0]->mutable_cpu_data()[0] = kld;  
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(KLDLayer);
REGISTER_LAYER_CLASS(KLD);

}  // namespace caffe

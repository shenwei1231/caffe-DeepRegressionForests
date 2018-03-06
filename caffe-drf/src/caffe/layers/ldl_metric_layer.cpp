#include <functional>
#include <utility>
#include <vector>
#include <cfloat>
#include "caffe/layers/ldl_metric_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void LDLMetricLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void LDLMetricLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void LDLMetricLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* pred = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  switch(this->layer_param_.ldl_metric_param().metric_type()) {
  case LDLMetricParameter_LDLMetricType_KLD: {
    // Kullback-Leibler Divergence
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
    break;
  } case LDLMetricParameter_LDLMetricType_Clark: {
    // Clark Distance
    Dtype clark = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      Dtype tmp = 0.0;
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        tmp += (q-p)*(q-p)/(q+p)/(q+p);
      }
      clark += sqrt(tmp);
    }
    clark = clark / N;
    top[0]->mutable_cpu_data()[0] = clark;
    break;
  } case LDLMetricParameter_LDLMetricType_Chebyshev: {
    // Chebyshev Distance
    Dtype cheby = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      Dtype tmp = 0.0;
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        if (tmp < fabs(q-p)) tmp = fabs(q-p);
      }
      cheby += tmp;
    }
    cheby = cheby / N;
    top[0]->mutable_cpu_data()[0] = cheby;
    break;
  } case LDLMetricParameter_LDLMetricType_Canberra: {
    // Canberra Metric
    Dtype canbe = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        canbe += fabs(q-p)/(q+p);
      }
    }
    canbe = canbe / N;
    top[0]->mutable_cpu_data()[0] = canbe;
    break;
  } case LDLMetricParameter_LDLMetricType_Cosine: {
    // Cosine Coefficient
    Dtype cos = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      Dtype tmp_qp = 0.0;
      Dtype tmp_pp = 0.0;
      Dtype tmp_qq = 0.0;
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        tmp_qp += q*p;
        tmp_qq += q*q;
        tmp_pp += p*p;
      }
      cos += tmp_qp/sqrt(tmp_qq)/sqrt(tmp_pp);
    }
    cos = cos / N;
    top[0]->mutable_cpu_data()[0] = cos;
    break;
  } case LDLMetricParameter_LDLMetricType_Inter: {
    // Intersection Similarity 
    Dtype inter = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        inter += std::min(q,p);
      }
    }
    inter = inter / N;
    top[0]->mutable_cpu_data()[0] = inter;
    break;
  }
  // Logistic Boosting Regression for Label Distribution Learning 
  case LDLMetricParameter_LDLMetricType_Fidelity: {
    // Fidelity
    Dtype fide = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        fide += sqrt(q*p);
      }
    }
    fide = fide / N;
    top[0]->mutable_cpu_data()[0] = fide;
    break;
  } case LDLMetricParameter_LDLMetricType_Euclid: {
    // Euclidean
    Dtype euc = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      Dtype tmp = 0.0;
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        tmp += (q-p)*(q-p);
      }
      euc += sqrt(tmp);
    }
    euc = euc / N;
    top[0]->mutable_cpu_data()[0] = euc;
    break;
  } case LDLMetricParameter_LDLMetricType_Soren: {
    // Sørensen
    Dtype soren = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      Dtype tmp_up = 0.0;
      Dtype tmp_down = 0.0;
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        tmp_up += fabs(q-p);
        tmp_down += q+p;
      }
      soren += tmp_up / tmp_down;
    }
    soren = soren / N;
    top[0]->mutable_cpu_data()[0] = soren;
    break;
  } case LDLMetricParameter_LDLMetricType_Square: {
    // SquaredX2
    Dtype square = 0.0;
    int N = bottom[0]->count(0,1);
    int D = bottom[0]->count(1);
    for (int n=0; n<N; ++n) {
      for (int d=0; d<D; ++d) {
        const Dtype p = std::max(Dtype(FLT_MIN), Dtype(pred[n*D+d]));
        const Dtype q = std::max(Dtype(FLT_MIN), Dtype(label[n*D+d]));
        square += (q-p)*(q-p)/(q+p);
      }
    }
    square = square / N;
    top[0]->mutable_cpu_data()[0] = square;
    break;
  }
  default:
    LOG(FATAL)<<"Unknown MetricType, should be 'KLD,Clark,Chebyshev,Canberra,Cosine,Inter,Intsim,Euclid,Soren,Square'";
  }
  
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(LDLMetricLayer);
REGISTER_LAYER_CLASS(LDLMetric);

}  // namespace caffe

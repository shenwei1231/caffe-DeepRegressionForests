#ifndef CAFFE_UTIL_SAMPLING_H_
#define CAFFE_UTIL_SAMPLING_H_

//#include "Eigen/Eigen"

namespace caffe {

template <typename Dtype>
void RandSample(int num_samples, int num_sub_samples, Dtype* sample_index_vec);

template <typename Dtype>
Dtype multivariate_gaussian(const Dtype* x, const Dtype* mu, const Dtype* sigma_square, int dim);



}


#endif
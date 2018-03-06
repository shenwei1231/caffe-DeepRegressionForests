/*
* @author Yilu Guo
 *
 *
 * Deep Regression Forests is open source code; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with Deep Regression Forests.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause 

 for more information.
*/

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/neural_decision_reg_forest_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/sampling.hpp"
#include "caffe/util/neural_decision_util_functions.hpp"
#include "caffe/util/benchmark.hpp"

#ifndef PI 
#define PI 3.1415926
#endif

#ifndef GPU_DEBUG 
#define GPU_DEBUG 0
#endif

namespace caffe
{
  
  /*
  template <typename Dtype>
  inline Dtype gaussian_1d(Dtype x, Dtype mu, Dtype sigma_square)
  {
    sigma_square = std::max(sigma_square, (Dtype) FLT_MIN);
    return (Dtype)1.0 / sqrt(2 * PI * sigma_square) * exp(-(x - mu) * (x - mu) / (2 * sigma_square));
  }
  template <typename Dtype>
  Dtype difference(const int n, Dtype* a, Dtype *b)
  {
      Dtype* c = new Dtype [n];
      caffe_sub(n, a, b, c);
      Dtype d = caffe_cpu_asum(n, c) / Dtype(n);
      delete []c;
      return d;
  }
  */

template <typename Dtype>
bool isdiff(Dtype x, Dtype y) {
  Dtype THRES = 0.000002;
  return std::abs(x - y) >= THRES;
}

__device__ int sub2ind_reg(int n, int c, int h, int w, int N, int C, int H, int W) {
  return  ((n * C + c) * H + h) * W + w;
}

__device__ int ind2sub_reg(int index, int C, int H, int W, int* n, int* c, int* h, int* w) {
  *w = index % W;
  *h = (index / W) % H;
  *c = (index / (W*H)) % C;
  *n = index / (C*W*H);
  return 0;
}

template <typename Dtype>
__device__ Dtype multivariate_gaussian_gpu(Dtype y, Dtype mu, Dtype sigma_square, int num_classes){
  return (float)1.0 / sqrt(2 * PI * (sigma_square + Dtype(FLT_MIN))) * (exp(-(y - mu) * (y - mu) / (2 * (sigma_square + Dtype(FLT_MIN))))+ Dtype(FLT_MIN));
}



template <typename Dtype>
__global__ void kernel_updata_all_reg(int num_outer, int num_inner,
          int num_trees, int num_leaf_nodes_per_tree, int num_classes, int iter_times, const Dtype* mu_all, const Dtype* sigma_all,
          Dtype const ** routing_vec, Dtype const ** label_vec, Dtype ** tree_prediction_vec) {
  int count = num_outer * num_inner * num_trees * iter_times;
  CUDA_KERNEL_LOOP(index, count) {
    int t, k, i, iter;
    int idx = index;
    ind2sub_reg(idx, num_outer, num_inner, num_trees, &iter, &i, &k, &t);
    const Dtype* label_data = label_vec[iter];
    Dtype* tree_prediction_prob_data = tree_prediction_vec[iter];
    const Dtype* routing_leaf_prob_data = routing_vec[iter];
    const Dtype y = label_data[sub2ind_reg(i, k, 0, 0, num_outer, num_inner, num_classes, 1)];
    for(int j = 0; j < num_leaf_nodes_per_tree; j++) {
      const Dtype mu = mu_all[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)];
      const Dtype sigma_square = sigma_all[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)];
      tree_prediction_prob_data[sub2ind_reg(i, k, t, 0, num_outer, num_inner, num_trees, num_classes)] +=
                routing_leaf_prob_data[sub2ind_reg(i, k, t, j, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree)] *
                max(multivariate_gaussian_gpu(y, mu, sigma_square, num_classes),Dtype(FLT_MIN));
    }
  }
}

template <typename Dtype> 
__global__ void kernel_mean_sig_reg(int num_trees, int num_leaf_nodes_per_tree, int num_outer, int num_inner, int iter_times,
    const Dtype* mu_all, const Dtype* sigma_all, Dtype const ** label_vec, Dtype const ** routing_vec, Dtype const ** tree_prediction_vec, 
    Dtype* mean_temp, Dtype* sigma_temp) {
    CUDA_KERNEL_LOOP(index, num_trees * num_leaf_nodes_per_tree) {
      int t, j;
      int idx = index;
      int num_classes = 1;
      j = idx % num_leaf_nodes_per_tree;
      t = idx / num_leaf_nodes_per_tree;
      Dtype zeta_sum = (Dtype) 0.0;
      Dtype mu_new;
      Dtype zeta;
      for (int iter = 0; iter < iter_times; iter++){
        const Dtype* label_data = label_vec[iter];
        const Dtype* tree_prediction_prob_data = tree_prediction_vec[iter];
        const Dtype* routing_leaf_prob_data = routing_vec[iter];
        for (int i = 0; i < num_outer; i++){
          for (int k = 0; k < num_inner; k++){
            const Dtype y = label_data[sub2ind_reg(i, k, 0, 0, num_outer, num_inner, num_classes, 1)];
            const Dtype mu = mu_all[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)];
            const Dtype sigma_square = sigma_all[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)];
            zeta = max(multivariate_gaussian_gpu(y, mu, sigma_square, num_classes), Dtype(FLT_MIN)) 
                * routing_leaf_prob_data[sub2ind_reg(i, k, t, j, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree)]
                / max(tree_prediction_prob_data[sub2ind_reg(i, k, t, 0, num_outer, num_inner, num_trees, num_classes)], Dtype(FLT_MIN));
            mean_temp[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)] += zeta*y;
            
            zeta_sum += zeta;
          }
        }
      }
      mean_temp[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)] *= (Dtype)1.0 / max(zeta_sum, Dtype(FLT_MIN));
      for (int iter = 0; iter < iter_times; iter++){
        const Dtype* label_data = label_vec[iter];
        const Dtype* tree_prediction_prob_data = tree_prediction_vec[iter];
        const Dtype* routing_leaf_prob_data = routing_vec[iter];
        for (int i = 0; i < num_outer; i++){
          for (int k = 0; k < num_inner; k++){
            const Dtype y = label_data[sub2ind_reg(i, k, 0, 0, num_outer, num_inner, num_classes, 1)];
            const Dtype mu = mu_all[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)];
            const Dtype sigma_square = sigma_all[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)];
            Dtype zeta= max(multivariate_gaussian_gpu(y, mu, sigma_square, num_classes), Dtype(FLT_MIN)) 
                * routing_leaf_prob_data[sub2ind_reg(i, k, t, j, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree)]
                / max(tree_prediction_prob_data[sub2ind_reg(i, k, t, 0, num_outer, num_inner, num_trees, num_classes)], Dtype(FLT_MIN));
            mu_new = mean_temp[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, 1)];
            mu_new = y - mu_new;
            sigma_temp[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, num_classes)] += zeta*mu_new*mu_new;
          }
        }
      }
      sigma_temp[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, num_classes)] *= (Dtype)1.0 / max(zeta_sum, Dtype(FLT_MIN));
      sigma_temp[sub2ind_reg(t, j, 0, 0, num_trees, num_leaf_nodes_per_tree, num_classes, num_classes)] += (Dtype) FLT_EPSILON;
    }
}

template <typename Dtype>
__global__ void kernel_backward_all_reg(Dtype* bottom_diff, Dtype* inter_data, const Dtype* tree_pred, const Dtype* mean_data,const Dtype* label_data, 
                                   const Dtype* routing_lf, const Dtype* dn_data, const Dtype* dim_offset,
                                   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
                                   int h, int w, int num_classes, int num_dims_, const Dtype scale_) {
  int num_nodes = num_split + num_leaf;
  CUDA_KERNEL_LOOP(index, num_outer) {
    for (int i=0; i<num_inner; ++i) {
      for(int t= 0; t < num_trees; ++t) {
        for (int l=0; l<num_leaf; ++l) {
          int inter_idx = sub2ind_reg(index,i,t,num_split+l, num_outer, num_inner, num_trees,num_nodes);
          int routing_lf_idx = sub2ind_reg(index, i, t, l, num_outer, num_inner, num_trees, num_leaf);
          for (int c=0; c<num_classes; ++c) {
            int lb_idx = sub2ind_reg(index,c,i/w,i%w, num_outer,num_classes,h,w);
            const Dtype label_value=label_data[lb_idx]/scale_;
            int tree_pred_idx = sub2ind_reg(index, i, t, c, num_outer, num_inner, num_trees, num_classes);
            int mean_idx = sub2ind_reg(t, l, c, 0, num_trees, num_leaf, num_classes, 1);
            inter_data[inter_idx] += (label_value - tree_pred[tree_pred_idx]) * mean_data[mean_idx];
          }
          inter_data[inter_idx] *= routing_lf[routing_lf_idx];
        }
      }
      for (int n=num_split-1; n>=0; --n) {
        for(int t = 0; t < num_trees; t++) {
          int dim_offset_idx = sub2ind_reg(t,n,0,0, num_trees,num_split,1,1);

          int diff_idx = sub2ind_reg(index,dim_offset[dim_offset_idx],i/w,i%w, num_outer,num_dims_,h,w);
          int inter_left_idx = sub2ind_reg(index,i,t,2*n+1,num_outer,num_inner,num_trees,num_nodes);
          int inter_right_idx = inter_left_idx + 1;
          bottom_diff[diff_idx] = (
                    dn_data[diff_idx] * inter_data[inter_right_idx] - 
                    (Dtype(1.0) - dn_data[diff_idx]) * inter_data[inter_left_idx]);
          int inter_parent_idx = sub2ind_reg(index,i,t,n,num_outer,num_inner,num_trees,num_nodes);
          inter_data[inter_parent_idx] = inter_data[inter_left_idx] + inter_data[inter_right_idx];

        }
      }
    }
  }
}


template <typename Dtype>
void NeuralDecisionRegForestWithLossLayer<Dtype>::UpdateTreePredictionAllDataGPU()
{
  if (num_classes_==1){
    int num_outer_iter = tree_prediction_all_data_prob_density_vec_[0].get()->shape(0);
    int num_inner_iter = tree_prediction_all_data_prob_density_vec_[0].get()->shape(1);

    Dtype ** tree_prediction_vec = new Dtype * [iter_times_in_epoch_];
    Dtype const ** routing_vec = new Dtype const * [iter_times_in_epoch_];
    Dtype const ** all_label_vec = new Dtype const * [iter_times_in_epoch_];
    for (int iter = 0; iter < iter_times_in_epoch_; iter++) {
      tree_prediction_vec[iter] = tree_prediction_all_data_prob_density_vec_[iter].get()->mutable_gpu_data();
      cudaMemset(tree_prediction_vec[iter], 0, sizeof(Dtype)* tree_prediction_all_data_prob_density_vec_[iter].get()->count());
      routing_vec[iter] = routing_leaf_all_data_prob_vec_[iter].get()->gpu_data();
      all_label_vec[iter] = all_data_label_vec_[iter].get()->gpu_data();
    }

    Dtype ** gpu_tree_prediction_vec;
    Dtype const ** gpu_routing_vec;
    Dtype const ** gpu_all_label_vec;
    cudaMalloc((void**)&gpu_tree_prediction_vec, sizeof(Dtype *)*iter_times_in_epoch_);
    cudaMemcpy(gpu_tree_prediction_vec, tree_prediction_vec, sizeof(Dtype *)*iter_times_in_epoch_, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_routing_vec, sizeof(Dtype const *)*iter_times_in_epoch_);
    cudaMemcpy(gpu_routing_vec, routing_vec, sizeof(Dtype const*)*iter_times_in_epoch_, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpu_all_label_vec, sizeof(Dtype const *)*iter_times_in_epoch_);
    cudaMemcpy(gpu_all_label_vec, all_label_vec, sizeof(Dtype const *)*iter_times_in_epoch_, cudaMemcpyHostToDevice);
    kernel_updata_all_reg<Dtype><<<CAFFE_GET_BLOCKS(num_outer_iter*num_inner_iter*iter_times_in_epoch_), CAFFE_CUDA_NUM_THREADS>>>(
              num_outer_iter, num_inner_iter, num_trees_, num_leaf_nodes_per_tree_, num_classes_,iter_times_in_epoch_,
              mean_->gpu_data(), sigma_square_->gpu_data(), gpu_routing_vec, gpu_all_label_vec, gpu_tree_prediction_vec);

  } else {
    for (int iter = 0; iter < iter_times_in_epoch_; iter++){
      Dtype* tree_prediction_all_data_prob_density_data = tree_prediction_all_data_prob_density_vec_[iter].get()->mutable_cpu_data();
      memset(tree_prediction_all_data_prob_density_data, 0, sizeof(Dtype)* tree_prediction_all_data_prob_density_vec_[iter].get()->count());

      const Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter].get()->cpu_data();

      int num_outer_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(0);
      int num_inner_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(1);
      for (int i = 0; i < num_outer_iter; i++){
        for (int k = 0; k < num_inner_iter; k++){
          const Dtype* y = all_data_label_vec_[iter].get()->cpu_data() + all_data_label_vec_[iter].get()->offset(i, k, 0, 0);
          for (int t = 0; t < num_trees_; t++){
            for (int j = 0; j < num_leaf_nodes_per_tree_; j++){
              tree_prediction_all_data_prob_density_data[tree_prediction_all_data_prob_density_vec_[iter].get()->offset(i, k, t, 0)] +=
                routing_leaf_all_data_prob_data[routing_leaf_all_data_prob_vec_[iter].get()->offset(i, k, t, j)] *
                max(multivariate_gaussian(y, mean_->cpu_data() + mean_->offset(t, j, 0, 0), sigma_square_->cpu_data() + sigma_square_->offset(t, j, 0, 0), num_classes_),Dtype(FLT_MIN));
            }
          }
        }
      }
    }
  }
}
template <typename Dtype>
void NeuralDecisionRegForestWithLossLayer<Dtype>::UpdateClassLabelDistrGPU() {
  num_epoch_++;
  LOG(INFO) << "Epoch " << num_epoch_ <<": Start updating class label distribution";
  int iter_times = 0;
  Dtype* mu_new = new Dtype [num_classes_];
  if (num_classes_==1){
    Blob<Dtype> mean_temp(mean_->shape());
    Dtype* mean_temp_data = mean_temp.mutable_gpu_data();
    Blob<Dtype> sigma_square_temp(sigma_square_->shape());
    Dtype* sigma_square_temp_data = sigma_square_temp.mutable_gpu_data();
    while (iter_times < iter_times_class_label_distr_) {
      LOG(INFO) << "Label distribution update iteration " << iter_times;
      UpdateTreePredictionAllDataGPU();
      cudaMemset(mean_temp.mutable_gpu_data(), 0, sizeof(Dtype)* mean_temp.count());
      cudaMemset(sigma_square_temp.mutable_gpu_data(), 0, sizeof(Dtype)* sigma_square_temp.count());

      Dtype const ** tree_prediction_vec = new Dtype const* [iter_times_in_epoch_];
      Dtype const ** routing_vec = new Dtype const * [iter_times_in_epoch_];
      Dtype const ** all_label_vec = new Dtype const * [iter_times_in_epoch_];
      int num_outer_iter = tree_prediction_all_data_prob_density_vec_[0].get()->shape(0);
      int num_inner_iter = tree_prediction_all_data_prob_density_vec_[0].get()->shape(1);
      for (int iter = 0; iter < iter_times_in_epoch_; iter++) {
        tree_prediction_vec[iter] = tree_prediction_all_data_prob_density_vec_[iter].get()->gpu_data();
        routing_vec[iter] = routing_leaf_all_data_prob_vec_[iter].get()->gpu_data();
        all_label_vec[iter] = all_data_label_vec_[iter].get()->gpu_data();
      }

      Dtype const ** gpu_tree_prediction_vec;
      Dtype const ** gpu_routing_vec;
      Dtype const ** gpu_all_label_vec;
      cudaMalloc((void**)&gpu_tree_prediction_vec, sizeof(Dtype const *)*iter_times_in_epoch_);
      cudaMemcpy(gpu_tree_prediction_vec, tree_prediction_vec, sizeof(Dtype const *)*iter_times_in_epoch_, cudaMemcpyHostToDevice);
      cudaMalloc((void**)&gpu_routing_vec, sizeof(Dtype const *)*iter_times_in_epoch_);
      cudaMemcpy(gpu_routing_vec, routing_vec, sizeof(Dtype const*)*iter_times_in_epoch_, cudaMemcpyHostToDevice);
      cudaMalloc((void**)&gpu_all_label_vec, sizeof(Dtype const *)*iter_times_in_epoch_);
      cudaMemcpy(gpu_all_label_vec, all_label_vec, sizeof(Dtype const *)*iter_times_in_epoch_, cudaMemcpyHostToDevice);

      kernel_mean_sig_reg<Dtype><<<CAFFE_GET_BLOCKS(num_trees_ * num_leaf_nodes_per_tree_), CAFFE_CUDA_NUM_THREADS>>>(
        num_trees_, num_leaf_nodes_per_tree_, num_outer_iter, num_inner_iter, iter_times_in_epoch_,
        mean_->gpu_data(), sigma_square_->gpu_data(), gpu_all_label_vec,
        gpu_routing_vec, gpu_tree_prediction_vec, mean_temp_data, sigma_square_temp_data);
      delete [] tree_prediction_vec; tree_prediction_vec=NULL;
      delete [] routing_vec; routing_vec=NULL;
      delete [] all_label_vec; all_label_vec=NULL;

      memcpy(mean_->mutable_cpu_data(), mean_temp.cpu_data(), sizeof(Dtype) * mean_->count());
      memcpy(sigma_square_->mutable_cpu_data(), sigma_square_temp.cpu_data(), sizeof(Dtype) * sigma_square_->count());
      iter_times++;
    }
  } else {
      Blob<Dtype> mean_temp(mean_->shape());
      Dtype* mean_temp_data = mean_temp.mutable_cpu_data();

      Blob<Dtype> sigma_square_temp(sigma_square_->shape());
      Dtype* sigma_square_temp_data = sigma_square_temp.mutable_cpu_data();
      while (iter_times < iter_times_class_label_distr_){
      LOG(INFO) << "Label distribution update iteration " << iter_times;
      UpdateTreePredictionAllData();
      memset(mean_temp_data, 0, sizeof(Dtype)* mean_temp.count());
      memset(sigma_square_temp_data, 0, sizeof(Dtype) * sigma_square_temp.count());
      of_ << "Iter " << iter_times <<":" << "\n";
      for (int t = 0; t < num_trees_; t++){
        for (int j = 0; j < num_leaf_nodes_per_tree_; j++){
          Dtype zeta_sum = (Dtype) 0.0;
          const Dtype* mu = mean_->cpu_data() + mean_->offset(t, j, 0, 0);
          const Dtype* sigma_square = sigma_square_->cpu_data() + sigma_square_->offset(t, j, 0, 0);

          for (int iter = 0; iter < iter_times_in_epoch_; iter++){
            int num_outer_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(0);
            int num_inner_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(1);
            
            for (int i = 0; i < num_outer_iter; i++){
              for (int k = 0; k < num_inner_iter; k++){
                const Dtype* y = all_data_label_vec_[iter].get()->cpu_data() + all_data_label_vec_[iter].get()->offset(i, k, 0, 0);
                Dtype zeta = max(multivariate_gaussian(y, mu, sigma_square, num_classes_), Dtype(FLT_MIN)) * routing_leaf_all_data_prob_vec_[iter].get()->data_at(i, k, t, j)
                  / max(tree_prediction_all_data_prob_density_vec_[iter].get()->data_at(i, k, t, 0), Dtype(FLT_MIN));
                caffe_axpy(num_classes_, zeta, y, mean_temp_data + mean_temp.offset(t, j, 0, 0));

                zeta_sum += zeta;
              }
            }
          }

          caffe_scal(num_classes_, (Dtype)1.0 / max(zeta_sum, Dtype(FLT_MIN)), mean_temp_data + mean_temp.offset(t, j, 0, 0));

          for (int iter = 0; iter < iter_times_in_epoch_; iter++){
            int num_outer_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(0);
            int num_inner_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(1);
            for (int i = 0; i < num_outer_iter; i++){
              for (int k = 0; k < num_inner_iter; k++){
                const Dtype* y = all_data_label_vec_[iter].get()->cpu_data() + all_data_label_vec_[iter].get()->offset(i, k, 0, 0);
                Dtype zeta = max(multivariate_gaussian(y, mu, sigma_square, num_classes_), Dtype(FLT_MIN)) * routing_leaf_all_data_prob_vec_[iter].get()->data_at(i, k, t, j)
                  / max(tree_prediction_all_data_prob_density_vec_[iter].get()->data_at(i, k, t, 0), Dtype(FLT_MIN));
                memcpy(mu_new, mean_temp_data + mean_temp.offset(t, j, 0, 0), sizeof(Dtype) * num_classes_);
                caffe_sub(num_classes_, y, mu_new, mu_new);
                caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_classes_, num_classes_, 1, zeta, mu_new, mu_new, (Dtype) 1.0, sigma_square_temp_data + sigma_square_temp.offset(t, j, 0, 0));
              }
            }
          }
          caffe_scal(num_classes_ * num_classes_, (Dtype)1.0 / max(zeta_sum, Dtype(FLT_MIN)), sigma_square_temp_data + sigma_square_temp.offset(t, j, 0, 0));
          caffe_add_scalar(num_classes_, (Dtype) FLT_EPSILON, sigma_square_temp_data + sigma_square_temp.offset(t, j, 0, 0));
        }
      }
      
      memcpy(mean_->mutable_cpu_data(), mean_temp_data, sizeof(Dtype) * mean_->count());
      memcpy(sigma_square_->mutable_cpu_data(), sigma_square_temp_data, sizeof(Dtype) * sigma_square_->count());
      iter_times++;
    }
  }
  LOG(INFO) << "Epoch" << num_epoch_ << ": End updating class label distribution";
  delete [] mu_new; mu_new = NULL;
  RecordClassLabelDistr();
}


  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    tree_for_training_ = caffe_rng_rand() % num_trees_;
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

    Dtype* routing_split_prob_data = routing_split_prob_.mutable_gpu_data();
    Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_gpu_data();
    const Dtype* dn_data = dn_->gpu_data();
    const Dtype* sub_dimensions_data = sub_dimensions_->gpu_data();

    kernel_routing<Dtype> <<<CAFFE_GET_BLOCKS(num_outer_ * num_inner_ * num_trees_),CAFFE_CUDA_NUM_THREADS >>>(
        num_outer_, num_trees_, num_dims_, bottom[0]->height(), bottom[0]->width(), num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_data, 
    sub_dimensions_data, routing_split_prob_data, routing_leaf_prob_data);

    const Dtype* mean_data = mean_->cpu_data();
    Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
    Dtype* all_data_label_data = all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
    Dtype* tree_prediction_data = tree_prediction_.mutable_cpu_data();
    caffe_set(tree_prediction_.count(), (Dtype) 0.0, tree_prediction_data);

    Dtype loss = (Dtype) 0.0;
    int count = 0;
    for (int i = 0; i < num_outer_; i++){
      for (int k = 0; k < num_inner_; k++){

        memcpy(routing_leaf_all_data_prob_data + routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, 0, 0),
        routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, 0, 0), sizeof(Dtype)* num_leaf_nodes_per_tree_ * num_trees_);

        if(drop_out_)
        {
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
              (Dtype)1.0, routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, tree_for_training_, 0),
              mean_data + mean_->offset(tree_for_training_, 0, 0, 0),
              (Dtype)0.0, tree_prediction_data + tree_prediction_.offset(i, k, tree_for_training_, 0));
        }
        else
        {
            for(int t = 0; t < num_trees_; t++)
            {
              caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
              (Dtype)1.0, routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, t, 0),
              mean_data + mean_->offset(t, 0, 0, 0),
              (Dtype)0.0, tree_prediction_data + tree_prediction_.offset(i, k, t, 0));
            }
        }

        for(int j = 0; j < num_classes_; ++j)
        {
          const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width())/scale_;
          all_data_label_data[all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, j, 0)]
          = label_value;

          if (drop_out_)
          {
            loss += 0.5 * (label_value - tree_prediction_.data_at(i, k, tree_for_training_, j)) * (label_value - tree_prediction_.data_at(i, k, tree_for_training_, j));
          }
          else
          {
            for(int t = 0; t < num_trees_; t++)
            {
              loss += 0.5 * (label_value - tree_prediction_.data_at(i, k, t, j)) * (label_value - tree_prediction_.data_at(i, k, t, j));
            }
          }
        }
        
        count++;
      }
    }

    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);

  }

  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {
    if (propagate_down[1])
    {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
      int count=0;
      if (drop_out_) {
        LOG(FATAL)<<"not implement";
        caffe_set(mean_->count(), static_cast<Dtype>(0), mean_->mutable_cpu_diff());
        caffe_set(sigma_square_->count(), static_cast<Dtype>(0), sigma_square_->mutable_cpu_diff());
        caffe_set(sub_dimensions_->count(), static_cast<Dtype>(0), sub_dimensions_->mutable_cpu_diff());
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
        
        Dtype* inter_var_data = inter_var_.mutable_cpu_data();
        memset(inter_var_data, (Dtype) 0.0, sizeof(Dtype) * inter_var_.count());
        const Dtype* dn_data = dn_->cpu_data();
        for (int i = 0; i < num_outer_; i++){
          for (int k = 0; k < num_inner_; k++){
            int t = tree_for_training_;{
              for (int l = 0; l < num_leaf_nodes_per_tree_; l++){
                for (int j = 0; j < num_classes_; j++){
                  const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width())/scale_;
                  inter_var_data[inter_var_.offset(i, k, t, num_split_nodes_per_tree_ + l)] += (label_value - tree_prediction_.data_at(i, k, t, j)) * mean_->data_at(t, l, j, 0);
                }
                inter_var_data[inter_var_.offset(i, k, t, num_split_nodes_per_tree_ + l)] *= routing_leaf_prob_.data_at(i, k, t, l);
              }
              for (int n = num_split_nodes_per_tree_ - 1; n >= 0; n--){
                int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
                bottom_diff[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] =
                  dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] * inter_var_.data_at(i, k, t, 2 * n + 2)
                  - ((Dtype)1.0 - dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())]) * inter_var_.data_at(i, k, t, 2 * n + 1);
                inter_var_data[inter_var_.offset(i, k, t, n)] = inter_var_.data_at(i, k, t, 2 * n + 2) + inter_var_.data_at(i, k, t, 2 * n + 1);
              }
              count++;  
            }
          }
        }
      } else {
        cudaMemset(mean_->mutable_gpu_diff(), 0, sizeof(Dtype)*mean_->count());
        cudaMemset(sigma_square_->mutable_gpu_diff(), 0, sizeof(Dtype)*sigma_square_->count());
        cudaMemset(sub_dimensions_->mutable_gpu_diff(), 0, sizeof(Dtype)*sub_dimensions_->count());
        cudaMemset(bottom[0]->mutable_gpu_diff(), 0, sizeof(Dtype)*bottom[0]->count());
        cudaMemset(inter_var_.mutable_gpu_data(), 0, sizeof(Dtype)*inter_var_.count());
        CHECK_EQ(dn_->width(), bottom[1]->width());
        CHECK_EQ(dn_->height(), bottom[1]->height());
        kernel_backward_all_reg<Dtype><<<CAFFE_GET_BLOCKS(num_outer_), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->mutable_gpu_diff(), inter_var_.mutable_gpu_data(),tree_prediction_.gpu_data(),mean_->gpu_data(),
        bottom[1]->gpu_data(), routing_leaf_prob_.gpu_data(), dn_->gpu_data(), sub_dimensions_->gpu_data(),
        num_outer_, num_inner_,num_trees_, num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_->height(),
        dn_->width(), num_classes_, num_dims_, scale_);
        count = num_outer_*num_inner_*num_trees_;
      }

      // Scale down gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(bottom[0]->count(), loss_weight / get_normalizer(normalization_, count), bottom[0]->mutable_cpu_diff());
    }
    if (iter_times_ && (iter_times_ + 1) % iter_times_in_epoch_ == 0) 
      UpdateClassLabelDistrGPU();
    iter_times_++;
  }
INSTANTIATE_LAYER_GPU_FUNCS(NeuralDecisionRegForestWithLossLayer);
}

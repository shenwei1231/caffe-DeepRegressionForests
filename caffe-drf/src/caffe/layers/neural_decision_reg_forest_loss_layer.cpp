/*
* @author Wei Shen
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

using namespace std;

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
  template < typename Dtype >
  void NeuralDecisionRegForestWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[0]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(dn_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

    NeuralDecisionForestParameter neural_decision_forest_param = this->layer_param_.neural_decision_forest_param();
    depth_ = neural_decision_forest_param.depth();
    num_trees_ = neural_decision_forest_param.num_trees();
    num_classes_ = neural_decision_forest_param.num_classes();
    iter_times_in_epoch_ = neural_decision_forest_param.iter_times_in_epoch();
    iter_times_class_label_distr_ = neural_decision_forest_param.iter_times_class_label_distr();
    all_data_vec_length_ = neural_decision_forest_param.all_data_vec_length();
    drop_out_ = neural_decision_forest_param.drop_out();

    scale_ = neural_decision_forest_param.scale();

    CHECK_GE(iter_times_in_epoch_, all_data_vec_length_);

    num_leaf_nodes_per_tree_ = (int)pow(2, depth_ - 1);
    num_split_nodes_per_tree_ = num_leaf_nodes_per_tree_ - 1;
    num_nodes_pre_tree_ = num_leaf_nodes_per_tree_ + num_split_nodes_per_tree_;


    num_dims_ = bottom[0]->shape(1);
    iter_times_ = 0;

    CHECK_LE(num_split_nodes_per_tree_, num_dims_)
    << "Number of the splitting nodes per tree must be less than the dimensions of the input feature";
    //num_classes_ = bottom[1]->shape(1);  //number of target channel for regression
    if (bottom[1]->num_axes()>1) CHECK_EQ(num_classes_, bottom[1]->shape(1)) 
      << "Assigned number of target channel for regression should equal to the channel number of label blob";
    num_nodes_ = num_trees_ * num_nodes_pre_tree_;

    
    this->blobs_.resize(3);
    this->blobs_[0].reset(new Blob<Dtype>(num_trees_, num_leaf_nodes_per_tree_, num_classes_, 1)); //mean
    mean_ = this->blobs_[0].get();
    Dtype* mean_data = mean_->mutable_cpu_data();
    for (int i = 0; i < mean_->count(); i++)
    {
      mean_data[i] = (Dtype) 1.0;//0.0
    }
    this->blobs_[1].reset(new Blob<Dtype>(num_trees_, num_leaf_nodes_per_tree_, num_classes_, num_classes_)); //std
    sigma_square_ = this->blobs_[1].get();
    Dtype* sigma_square_data = sigma_square_->mutable_cpu_data();
    for (int i = 0; i < sigma_square_->count(); i++)
    {
      sigma_square_data[i] = (Dtype) 1.0;
    }

    if_.open(neural_decision_forest_param.init_filename().c_str(), ifstream::binary | ifstream::in);
    if(if_.is_open())
    {
      if_.seekg(0);
      if_.read((char*)mean_data, sizeof(Dtype) * mean_->count());
      if_.read((char*)sigma_square_data, sizeof(Dtype) * sigma_square_->count());
    }
    if_.close();
    for (int i = 0; i < mean_->count(); i++)
    {
      mean_data[i] = mean_data[i]/scale_;
    }
    // if (num_classes_>1){
    //     for (int i = 1; i < sigma_square_->count(); i=i+4)
    //     {
    //         sigma_square_data[i] = (Dtype) 0.0;
    //         sigma_square_data[i+1] = (Dtype) 0.0;
    //     }
    // }
    this->blobs_[2].reset(new Blob<Dtype>(num_trees_, num_split_nodes_per_tree_, 1, 1));
    sub_dimensions_ = this->blobs_[2].get();
    Dtype* sub_dimensions_data = sub_dimensions_->mutable_cpu_data();
    for (int t = 0; t < num_trees_; ++t) 
      RandSample(num_dims_, num_split_nodes_per_tree_, sub_dimensions_data + sub_dimensions_->offset(t, 0, 0, 0));

    routing_leaf_all_data_prob_vec_.resize(iter_times_in_epoch_);
    tree_prediction_all_data_prob_density_vec_.resize(iter_times_in_epoch_);
    all_data_label_vec_.resize(iter_times_in_epoch_);


    of_.open(neural_decision_forest_param.record_filename().c_str());
    

    tree_for_training_ = 0;

    num_epoch_ = 0;
    drop_out_ = false;

    b_distr_updated_ = true;
    loss_type_ = 0;
    iter_time_update_stop_ = 0;
  }

  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    LossLayer<Dtype>::Reshape(bottom, top);
    sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
    axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.neural_decision_forest_param().axis());
    num_outer_ = bottom[0]->count(0, axis_);
    num_inner_ = bottom[0]->count(axis_ + 1);
    if (bottom[0]->num_axes()>2) CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
    if (bottom[0]->num_axes()>3) CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3));
    /*
    CHECK_EQ(num_outer_ * num_inner_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, 1, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with continuous values in [0, 1].";
    */
    routing_split_prob_.Reshape(num_outer_, num_inner_, num_trees_, num_split_nodes_per_tree_);
    InitRoutingProb();
    routing_leaf_prob_.Reshape(num_outer_, num_inner_, num_trees_, num_leaf_nodes_per_tree_);

    tree_prediction_prob_density_.Reshape(num_outer_, num_inner_, num_trees_, 1);

    tree_prediction_.Reshape(num_outer_, num_inner_, num_trees_, num_classes_);

    inter_var_.Reshape(num_outer_, num_inner_, num_trees_, num_nodes_pre_tree_);

    iter_time_current_epcho_ = iter_times_ % iter_times_in_epoch_;

    routing_leaf_all_data_prob_vec_[iter_time_current_epcho_].reset(new Blob<Dtype>(num_outer_, num_inner_, num_trees_, num_leaf_nodes_per_tree_));

    tree_prediction_all_data_prob_density_vec_[iter_time_current_epcho_].reset(new Blob<Dtype>(num_outer_, num_inner_, num_trees_, num_classes_));

    all_data_label_vec_[iter_time_current_epcho_].reset(new Blob<Dtype>(num_outer_, num_inner_, num_classes_, 1));

    normalization_ = LossParameter_NormalizationMode_VALID;

  }

  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    tree_for_training_ = caffe_rng_rand() % num_trees_;
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

    Dtype* routing_split_prob_data = routing_split_prob_.mutable_cpu_data();
    Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_cpu_data();
    Dtype* tree_prediction_data = tree_prediction_.mutable_cpu_data();
    caffe_set(tree_prediction_.count(), (Dtype) 0.0, tree_prediction_data);
    
    const Dtype* mean_data = mean_->cpu_data();

    Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
    Dtype* all_data_label_data = all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();

    Dtype loss = (Dtype) 0.0;
    int count = 0;
    for (int i = 0; i < num_outer_; i++)
    {
      for (int k = 0; k < num_inner_; k++)
      {
        for (int t = 0; t < num_trees_; ++t)
        {
          for (int n = 0; n < num_split_nodes_per_tree_; ++n)
          {
            int current_offset = n;
            int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
            int left_child_offset = 2 * current_offset + 1;
            int right_child_offset = 2 * current_offset + 2;
            if (right_child_offset < num_split_nodes_per_tree_)
            {
              routing_split_prob_data[routing_split_prob_.offset(i, k, t, left_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width());
              routing_split_prob_data[routing_split_prob_.offset(i, k, t, right_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * ((Dtype) 1.0 - dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width()));
            }
            else
            {
              left_child_offset -= num_split_nodes_per_tree_;
              right_child_offset -= num_split_nodes_per_tree_;
              routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, t, left_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width());
              routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, t, right_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * ((Dtype) 1.0 - dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width()));
            }
          }
        }

        memcpy(routing_leaf_all_data_prob_data + routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, 0, 0),
        routing_leaf_prob_data + routing_leaf_prob_.offset(i, k, 0, 0), sizeof(Dtype)* num_leaf_nodes_per_tree_ * num_trees_);

        if(drop_out_)
        {
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
              (Dtype)1.0, routing_leaf_prob_data + routing_leaf_prob_.offset(i, k, tree_for_training_, 0),
              mean_data + mean_->offset(tree_for_training_, 0, 0, 0),
              (Dtype)0.0, tree_prediction_data + tree_prediction_.offset(i, k, tree_for_training_, 0));
        }
        else
        {
            for(int t = 0; t < num_trees_; t++)
            {
              caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
              (Dtype)1.0, routing_leaf_prob_data + routing_leaf_prob_.offset(i, k, t, 0),
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
  void NeuralDecisionRegForestWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {
    if (propagate_down[1])
    {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
      caffe_set(mean_->count(), static_cast<Dtype>(0), mean_->mutable_cpu_diff());
      caffe_set(sigma_square_->count(), static_cast<Dtype>(0), sigma_square_->mutable_cpu_diff());
      caffe_set(sub_dimensions_->count(), static_cast<Dtype>(0), sub_dimensions_->mutable_cpu_diff());
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
      
      Dtype* inter_var_data = inter_var_.mutable_cpu_data();
      memset(inter_var_data, (Dtype) 0.0, sizeof(Dtype) * inter_var_.count());
      const Dtype* dn_data = dn_->cpu_data();
      int count = 0;
      
      for (int i = 0; i < num_outer_; i++)
      {
        for (int k = 0; k < num_inner_; k++)
        {
          if (drop_out_)
          {
            int t = tree_for_training_;
            {
            
              for (int l = 0; l < num_leaf_nodes_per_tree_; l++)
              {
                  for (int j = 0; j < num_classes_; j++)
                  {
                    const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width())/scale_;
                    inter_var_data[inter_var_.offset(i, k, t, num_split_nodes_per_tree_ + l)] += (label_value - tree_prediction_.data_at(i, k, t, j)) * mean_->data_at(t, l, j, 0);
                  }
                  inter_var_data[inter_var_.offset(i, k, t, num_split_nodes_per_tree_ + l)] *= routing_leaf_prob_.data_at(i, k, t, l);
              }
              for (int n = num_split_nodes_per_tree_ - 1; n >= 0; n--)
              {
                int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
                bottom_diff[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] =
                  dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] * inter_var_.data_at(i, k, t, 2 * n + 2)
                  - ((Dtype)1.0 - dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())]) * inter_var_.data_at(i, k, t, 2 * n + 1);
                inter_var_data[inter_var_.offset(i, k, t, n)] = inter_var_.data_at(i, k, t, 2 * n + 2) + inter_var_.data_at(i, k, t, 2 * n + 1);
              }
              count++;
              
            }
          }
          else
          {
            for(int t = 0; t < num_trees_; t++)
            {
            
              for (int l = 0; l < num_leaf_nodes_per_tree_; l++)
              {
                  for (int j = 0; j < num_classes_; j++)
                  {
                    const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width())/scale_;
                    inter_var_data[inter_var_.offset(i, k, t, num_split_nodes_per_tree_ + l)] += (label_value - tree_prediction_.data_at(i, k, t, j)) * mean_->data_at(t, l, j, 0);
                  }
                  inter_var_data[inter_var_.offset(i, k, t, num_split_nodes_per_tree_ + l)] *= routing_leaf_prob_.data_at(i, k, t, l);
              }
              for (int n = num_split_nodes_per_tree_ - 1; n >= 0; n--)
              {
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

      }
      // Scale down gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(bottom[0]->count(), loss_weight / get_normalizer(normalization_, count), bottom_diff);
    }
    if (iter_times_ && (iter_times_ + 1) % iter_times_in_epoch_ == 0) 
      UpdateClassLabelDistr();
    iter_times_++;
  }

  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::InitRoutingProb()
  {
    Dtype* routing_split_prob_data = routing_split_prob_.mutable_cpu_data();
    for (int i = 0; i < num_outer_; i++)
    {
      for (int j = 0; j < num_inner_; j++)
      {
        for (int t = 0; t < num_trees_; t++)
        {
          routing_split_prob_data[routing_split_prob_.offset(i, j, t, 0)] = (Dtype) 1.0;
        }
      }
    }
  }
    template <typename Dtype>
  Dtype NeuralDecisionRegForestWithLossLayer<Dtype>::get_normalizer(LossParameter_NormalizationMode normalization_mode, int valid_count)
  {
    Dtype normalizer;
    switch (normalization_mode) 
    {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(num_outer_ * num_inner_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(num_outer_ * num_inner_);
      }
      else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(num_outer_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
        << LossParameter_NormalizationMode_Name(normalization_mode);
    }
    // Some users will have no labels for some examples in order to 'turn off' a
    // particular loss in a multi-task setup. The max prevents NaNs in that case.
    return max(Dtype(1.0), normalizer);
  }

  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::UpdateTreePredictionAllData()
  {
    
    for (int iter = 0; iter < iter_times_in_epoch_; iter++)
    {
      Dtype* tree_prediction_all_data_prob_density_data = tree_prediction_all_data_prob_density_vec_[iter].get()->mutable_cpu_data();
      memset(tree_prediction_all_data_prob_density_data, 0, sizeof(Dtype)* tree_prediction_all_data_prob_density_vec_[iter].get()->count());

      const Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter].get()->cpu_data();

      int num_outer_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(0);
      int num_inner_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(1);

      for (int i = 0; i < num_outer_iter; i++)
      {
        for (int k = 0; k < num_inner_iter; k++)
        {

          const Dtype* y = all_data_label_vec_[iter].get()->cpu_data() + all_data_label_vec_[iter].get()->offset(i, k, 0, 0);
          for (int t = 0; t < num_trees_; t++)
          {
            for (int j = 0; j < num_leaf_nodes_per_tree_; j++)
            {
              tree_prediction_all_data_prob_density_data[tree_prediction_all_data_prob_density_vec_[iter].get()->offset(i, k, t, 0)] +=
                routing_leaf_all_data_prob_data[routing_leaf_all_data_prob_vec_[iter].get()->offset(i, k, t, j)] *
                max(multivariate_gaussian(y, mean_->cpu_data() + mean_->offset(t, j, 0, 0), sigma_square_->cpu_data() + sigma_square_->offset(t, j, 0, 0), num_classes_),Dtype(FLT_MIN));
            }
          }

        }

      }
    }
  }


  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::UpdateClassLabelDistr()
  {
    
    
    
      num_epoch_++;
      LOG(INFO) << "Epoch " << num_epoch_ << ": Start updating class label distribution";
      of_ << "------------------Epoch " << num_epoch_ << " ------------------" << "\n";
      Blob<Dtype> mean_temp(mean_->shape());
      Dtype* mean_temp_data = mean_temp.mutable_cpu_data();

      Blob<Dtype> sigma_square_temp(sigma_square_->shape());
      Dtype* sigma_square_temp_data = sigma_square_temp.mutable_cpu_data();

      Dtype* mu_new = new Dtype [num_classes_];

      int iter_times = 0;
      while (iter_times < iter_times_class_label_distr_)
      {
        LOG(INFO) << "Label distribution update iteration " << iter_times;
        UpdateTreePredictionAllData();
        memset(mean_temp_data, 0, sizeof(Dtype)* mean_temp.count());
        memset(sigma_square_temp_data, 0, sizeof(Dtype) * sigma_square_temp.count());
        of_ << "Iter " << iter_times <<":" << "\n";
        for (int t = 0; t < num_trees_; t++)
        {
          for (int j = 0; j < num_leaf_nodes_per_tree_; j++)
          {
            Dtype zeta_sum = (Dtype) 0.0;
            const Dtype* mu = mean_->cpu_data() + mean_->offset(t, j, 0, 0);
            const Dtype* sigma_square = sigma_square_->cpu_data() + sigma_square_->offset(t, j, 0, 0);
            for (int iter = 0; iter < iter_times_in_epoch_; iter++)
            {
              int num_outer_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(0);
              int num_inner_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(1);
              
              for (int i = 0; i < num_outer_iter; i++)
              {
                for (int k = 0; k < num_inner_iter; k++)
                {
                  
                  const Dtype* y = all_data_label_vec_[iter].get()->cpu_data() + all_data_label_vec_[iter].get()->offset(i, k, 0, 0);

                  Dtype zeta = max(multivariate_gaussian(y, mu, sigma_square, num_classes_), Dtype(FLT_MIN)) * routing_leaf_all_data_prob_vec_[iter].get()->data_at(i, k, t, j)
                    / max(tree_prediction_all_data_prob_density_vec_[iter].get()->data_at(i, k, t, 0), Dtype(FLT_MIN));
                  caffe_axpy(num_classes_, zeta, y, mean_temp_data + mean_temp.offset(t, j, 0, 0));

                  zeta_sum += zeta;
                  
                }
              }
            }

            caffe_scal(num_classes_, (Dtype)1.0 / max(zeta_sum, Dtype(FLT_MIN)), mean_temp_data + mean_temp.offset(t, j, 0, 0));
            
            for (int iter = 0; iter < iter_times_in_epoch_; iter++)
            {
              int num_outer_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(0);
              int num_inner_iter = tree_prediction_all_data_prob_density_vec_[iter].get()->shape(1);
              for (int i = 0; i < num_outer_iter; i++)
              {
                for (int k = 0; k < num_inner_iter; k++)
                {
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
      LOG(INFO) << "Epoch" << num_epoch_ << ": End updating class label distribution";
      delete [] mu_new; mu_new = NULL;
      RecordClassLabelDistr();
  }
  template <typename Dtype>
  void NeuralDecisionRegForestWithLossLayer<Dtype>::RecordClassLabelDistr()
  {
    of_ << "Epoch: " << num_epoch_ << "\n";
    for (int t = 0; t < num_trees_; t++)
    {
      of_ << "tree " << t << "\n";
      for (int i = 0; i < num_leaf_nodes_per_tree_; i++)
      {
        of_ << "  leaf node_" << i << "\n";
        of_ << "mu: " << mean_->data_at(t, i, 0, 0) << " sigma: " << sigma_square_->data_at(t, i, 0, 0) << "\n";
      }
    }

  }

#ifdef CPU_ONLY
  STUB_GPU(NeuralDecisionRegForestWithLossLayer);
#endif

  INSTANTIATE_CLASS(NeuralDecisionRegForestWithLossLayer);
  REGISTER_LAYER_CLASS(NeuralDecisionRegForestWithLoss);
}

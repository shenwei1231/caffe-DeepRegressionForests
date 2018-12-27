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

#include "caffe/layers/neural_decision_reg_forest_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
  template < typename Dtype >
  void NeuralDecisionRegForestLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[0]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(dn_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

    NeuralDecisionForestParameter neural_decision_forest_param = this->layer_param_.neural_decision_forest_param();
    depth_ = neural_decision_forest_param.depth();
    num_trees_ = neural_decision_forest_param.num_trees();
    num_classes_ = neural_decision_forest_param.num_classes();
    scale_ = neural_decision_forest_param.scale();

    num_leaf_nodes_per_tree_ = (int)pow(2, depth_ - 1);
    num_split_nodes_per_tree_ = num_leaf_nodes_per_tree_ - 1;
    num_nodes_pre_tree_ = num_leaf_nodes_per_tree_ + num_split_nodes_per_tree_;


    num_dims_ = bottom[0]->shape(1);
    

    CHECK_LE(num_split_nodes_per_tree_, num_dims_)
    << "Number of the splitting nodes per tree must be less than the dimensions of the input feature";
    //num_classes_ = bottom[1]->shape(1);  //number of target channel for regression

    num_nodes_ = num_trees_ * num_nodes_pre_tree_;

    

    this->blobs_.resize(3);
    this->blobs_[0].reset(new Blob<Dtype>(num_trees_, num_leaf_nodes_per_tree_, num_classes_, 1)); //mean
    mean_ = this->blobs_[0].get();
    
    this->blobs_[1].reset(new Blob<Dtype>(num_trees_, num_leaf_nodes_per_tree_, num_classes_, num_classes_)); //std
    sigma_square_ = this->blobs_[1].get();
    

    
    this->blobs_[2].reset(new Blob<Dtype>(num_trees_, num_split_nodes_per_tree_, 1, 1));
    sub_dimensions_ = this->blobs_[2].get();
    
    
  }

  template <typename Dtype>
	void NeuralDecisionRegForestLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.neural_decision_forest_param().axis());
		num_outer_ = bottom[0]->count(0, axis_);
		num_inner_ = bottom[0]->count(axis_ + 1);
		routing_split_prob_.Reshape(num_outer_, num_inner_, num_trees_, num_split_nodes_per_tree_);
		InitRoutingProb();
		routing_leaf_prob_.Reshape(num_outer_, num_inner_, num_trees_, num_leaf_nodes_per_tree_);
		forest_prediction_.Reshape(num_outer_, num_inner_, num_classes_, 1);
		top[0]->Reshape(num_outer_, num_classes_, bottom[0]->height(), bottom[0]->width());
	}


	template <typename Dtype>
	void NeuralDecisionRegForestLayer<Dtype>::InitRoutingProb()
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
	void NeuralDecisionRegForestLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
		//const Dtype* dn_data = dn_->cpu_data();
		Dtype* routing_split_prob_data = routing_split_prob_.mutable_cpu_data();
		Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_cpu_data();

		const Dtype* mean_data = mean_->cpu_data();
		Dtype* forest_prediction_data = forest_prediction_.mutable_cpu_data();
		//const Dtype* label = bottom[1]->cpu_data();

		
		const Blob<Dtype>* output_ = top[0];
		Dtype* output_data = top[0]->mutable_cpu_data();
		memset(output_data, 0, sizeof(Dtype) * output_->count(0));
		for (int i = 0; i < num_outer_; i++)
		{
			for (int k = 0; k < num_inner_; k++)
			{
				for (int t = 0; t < num_trees_; ++t)
				{
					for (int j = 0; j < num_split_nodes_per_tree_; ++j)
					{
						int current_offset = j;
						int dim_offset = (int)sub_dimensions_->data_at(t, j, 0, 0);
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
							//LOG(INFO) << routing_leaf_prob_data << routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, tree_index, left_child_offset)] << ", " << routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, tree_index, right_child_offset)] << endl;
						}
					}
				}
			
				//for (int t = 0; t < num_trees_; t++)
				{

					caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_trees_ * num_leaf_nodes_per_tree_,
						(Dtype)1.0, routing_leaf_prob_data + routing_leaf_prob_.offset(i, k, 0, 0),
						mean_data, (Dtype)0.0, forest_prediction_data + forest_prediction_.offset(i, k, 0, 0));

					for (int c = 0; c < num_classes_; c++)
					{
						output_data[output_->offset(i, c, k / dn_->width(), k % dn_->width())]
							= forest_prediction_data[forest_prediction_.offset(i, k, c, 0)];
						//LOG(INFO) << "output_prob[" << i << " " << k << " " << c << "]:" << forest_prediction_prob_data[forest_prediction_prob_.offset(i, k, c, 0)] << endl;
					}

				}
				
					/*caffe_add(num_classes_, tree_prediction_prob_data + tree_prediction_prob_.offset(i, k, t, 0),
						forest_prediction_prob_data + forest_prediction_prob_->offset(i, k, 0, 0),
						forest_prediction_prob_data + forest_prediction_prob_->offset(i, k, 0, 0));*/
				
			}
		}
		/*BlobProto blob_proto;
		routing_leaf_prob_.ToProto(&blob_proto);
		WriteProtoToTextFile(blob_proto, "routing_leaf_cpu.txt");
		routing_split_prob_.ToProto(&blob_proto);
		WriteProtoToTextFile(blob_proto, "routing_split_cpu.txt");*/

		caffe_scal(output_->count(), scale_ / num_trees_, output_data);
	}

	template <typename Dtype>
	void NeuralDecisionRegForestLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{

	}
	
	#ifdef CPU_ONLY
	  STUB_GPU(NeuralDecisionRegForestLayer);
	#endif
    INSTANTIATE_CLASS(NeuralDecisionRegForestLayer);
    REGISTER_LAYER_CLASS(NeuralDecisionRegForest);
}

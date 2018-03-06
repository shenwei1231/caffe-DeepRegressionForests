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
#include "caffe/util/neural_decision_util_functions.hpp"

namespace caffe
{
	
	template <typename Dtype>
	void NeuralDecisionRegForestLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Blob<Dtype> * output_ = top[0];
		Dtype* output_data = output_->mutable_gpu_data();
		

		sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
		const Dtype* dn_data = dn_->gpu_data();
		Dtype* routing_split_prob_data = routing_split_prob_.mutable_gpu_data();
		Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_gpu_data();

		const Dtype* mean_data = mean_->gpu_data();
		const Dtype* sub_dimensions_data = sub_dimensions_->gpu_data();
		
		
		Dtype* forest_prediction_data = forest_prediction_.mutable_gpu_data();

		kernel_routing<Dtype> << <CAFFE_GET_BLOCKS(num_outer_ * num_inner_ * num_trees_),
			CAFFE_CUDA_NUM_THREADS >> >(num_outer_, num_trees_, num_dims_, bottom[0]->height(), bottom[0]->width(), num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_data, sub_dimensions_data,
			routing_split_prob_data, routing_leaf_prob_data);
		
	
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_outer_ * num_inner_, num_classes_, num_trees_ * num_leaf_nodes_per_tree_,
			(Dtype)1.0, routing_leaf_prob_data, mean_data, (Dtype)0.0, forest_prediction_data);

		caffe_gpu_scal(num_outer_ * num_inner_ * num_classes_, scale_ / num_trees_, forest_prediction_data);
		//caffe_gpu_scal(num_outer_ * num_inner_ * num_classes_, (Dtype)1.0 / num_trees_, forest_prediction_data);
		CHECK_EQ(forest_prediction_.count(), output_->count());

		kernel_transform<Dtype> << < CAFFE_GET_BLOCKS(num_outer_ * num_inner_ * num_classes_),
			CAFFE_CUDA_NUM_THREADS >> > (num_outer_, num_classes_, bottom[0]->height(), bottom[0]->width(),
			forest_prediction_data, output_data);
        //Dtype* output_cpu = output_->mutable_cpu_data();
        //LOG(INFO)<<output_cpu[output_->offset(0,0,0,0)];
//        Dtype* output_cpu = output_->mutable_cpu_data();
//        for (int i=0;i<output_->count();i++){
//            output_cpu[i]=round(output_cpu[i]);
//        }
	}

	template <typename Dtype>
	void NeuralDecisionRegForestLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Backward_cpu(top, propagate_down, bottom);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NeuralDecisionRegForestLayer);
}

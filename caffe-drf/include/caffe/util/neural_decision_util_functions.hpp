/*
* @author Wei Shen
 *
 *
 * LDLForest is open source code; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with LDLForest .  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause 

 for more information.
*/

#ifndef CAFFE_NEURAL_DECISION_UTIL_FUNCTIONS_
#define CAFFE_NEURAL_DECISION_UTIL_FUNCTIONS_

namespace caffe
{
	__device__ int sub2ind(int n, int c, int h, int w, int N, int C, int H, int W);

	__device__ int ind2sub(int index, int C, int H, int W, int* n, int* c, int* h, int* w);

	template <typename Dtype>
	__global__ void kernel_routing(const int num, const int trees, const int dn_channel_dim,
		const int spatial_h_dim, const int spatial_w_dim, const int leaf_nodes_per_tree, 
		const int split_nodes_pre_tree, const Dtype* dn_data, const Dtype* sub_dim_data, Dtype* routing_split_out, Dtype* routing_leaf_out);

	template <typename Dtype>
	__global__ void kernel_transform(const int num, const int channel,
		const int height, const int width, Dtype* prediction_in, Dtype* prediction_out);

	template <typename Dtype>
	__global__ void kernel_updata_all(int num_outer_iter, int num_inner_iter,
          int num_trees, int num_leaf_nodes_per_tree, int num_class, const Dtype* routing_data, 
          const Dtype* class_label_distr_data, Dtype* pred_data);

	template <typename Dtype> 
	__global__ void kernel_update_leaf(int num_trees, int num_leaf_nodes_per_tree, int num_class, int num_outer, int num_inner,
    const Dtype* class_label_distr_data, const Dtype* label_data, const Dtype* routing_leaf_prob_data, const Dtype* tree_prediction_prob_data, 
    Dtype* class_label_distr_temp_data);

	/*
    template <typename Dtype>
	__global__ void kernel_bottom_diff(int num_outer, int num_inner, int num_trees, int num_leaf_nodes_per_tree,
	  int num_split_nodes_per_tree, int num_nodes_pre_tree, int num_class, int t, int N, int C, int H, int W,
	  const Dtype* class_label_distr_data, const Dtype* routing_leaf_data,
	  const Dtype* dn_data, const Dtype* sub_dim_data, const Dtype* tree_prediction_data, 
	  const Dtype* label_data, Dtype* inter_var_data, Dtype* bottom_diff);
	*/

	template <typename Dtype>
	__global__ void kernel_backward(Dtype* bottom_diff, Dtype* inter_data, const Dtype* cls_lb_distr, const Dtype* label_data, 
   const Dtype* routing_lf, const Dtype* dn_data, const Dtype* tree_pred, const Dtype* dim_offset,
   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
   int h, int w, int num_classes, int tree_id, int num_dims_);


	template <typename Dtype>
	__global__ void kernel_backward_all(Dtype* bottom_diff, Dtype* inter_data, const Dtype* cls_lb_distr, const Dtype* label_data, 
   const Dtype* routing_lf, const Dtype* dn_data, const Dtype* tree_pred, const Dtype* dim_offset,
   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
   int h, int w, int num_classes, int num_dims_); 

}


#endif
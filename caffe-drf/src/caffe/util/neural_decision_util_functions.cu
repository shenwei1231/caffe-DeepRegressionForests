/*
* @author Wei Shen
 *
 *
 * Regression Forest is open source code; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with Regression Forest .  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause 

 for more information.
*/

#include <float.h>
#include "caffe/util/neural_decision_util_functions.hpp"
#include "caffe/common.hpp"

using namespace std;

namespace caffe
{
	__device__ int sub2ind(int n, int c, int h, int w, int N, int C, int H, int W) 
	{
	  	return  ((n * C + c) * H + h) * W + w;
	}	

	__device__ int ind2sub(int index, int C, int H, int W, int* n, int* c, int* h, int* w) 
	{
	  *w = index % W;
	  *h = (index / W) % H;
	  *c = (index / (W*H)) % C;
	  *n = index / (C*W*H);
	  return 0;
	}

	template <>
	__global__ void kernel_routing<float>(const int num, const int trees, const int dn_channel_dim,
		const int spatial_h_dim, const int spatial_w_dim, const int leaf_nodes_per_tree, 
		const int split_nodes_pre_tree, const float* dn_data, const float* sub_dim_data, float* routing_split_out, float* routing_leaf_out) 
	{
		int spatial_dim = spatial_h_dim * spatial_w_dim;
		CUDA_KERNEL_LOOP(index, num * spatial_dim * trees) 
		{
			int n, s, t, j;
			int idx = index;
			
			ind2sub(idx, spatial_dim, trees, 1, &n, &s, &t, &j);
			
			for (int current_offset = 0; current_offset < split_nodes_pre_tree; current_offset++)
			{
				int left_child_offset = 2 * current_offset + 1;
				int right_child_offset = 2 * current_offset + 2;
				
				int sub_dim_offset = (int) sub_dim_data[sub2ind(t, current_offset, 0, 0, trees, split_nodes_pre_tree, 1, 1)];

				float dn = dn_data[sub2ind(n, sub_dim_offset, s, 0, num, dn_channel_dim, spatial_dim, 1)];
				if (right_child_offset < split_nodes_pre_tree)
				{
					routing_split_out[sub2ind(n, s, t, left_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] 
					= routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;

					routing_split_out[sub2ind(n, s, t, right_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] = 
					routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((float) 1.0 - dn);
				}
				else
				{
					right_child_offset -= split_nodes_pre_tree;
					left_child_offset -= split_nodes_pre_tree;
					routing_leaf_out[sub2ind(n, s, t, left_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
					= routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;
					routing_leaf_out[sub2ind(n, s, t, right_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
					= routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((float) 1.0 - dn);
				}
			}
		}
	}


	template <>
	__global__ void kernel_routing<double>(const int num, const int trees, const int dn_channel_dim,
		const int spatial_h_dim, const int spatial_w_dim, const int leaf_nodes_per_tree, 
		const int split_nodes_pre_tree, const double* dn_data, const double* sub_dim_data, double* routing_split_out, double* routing_leaf_out) 
	{
		int spatial_dim = spatial_h_dim * spatial_w_dim;
		CUDA_KERNEL_LOOP(index, num * spatial_dim * trees) 
		{
			int n, s, t, j;
			int idx = index;
			
			ind2sub(idx, spatial_dim, trees, 1, &n, &s, &t, &j);
			
			for (int current_offset = 0; current_offset < split_nodes_pre_tree; current_offset++)
			{
				int left_child_offset = 2 * current_offset + 1;
				int right_child_offset = 2 * current_offset + 2;
				
				int sub_dim_offset = (int) sub_dim_data[sub2ind(t, current_offset, 0, 0, trees, split_nodes_pre_tree, 1, 1)];

				double dn = dn_data[sub2ind(n, sub_dim_offset, s, 0, num, dn_channel_dim, spatial_dim, 1)];
				if (right_child_offset < split_nodes_pre_tree)
				{
					routing_split_out[sub2ind(n, s, t, left_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] 
					= routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;

					routing_split_out[sub2ind(n, s, t, right_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] = 
					routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((double) 1.0 - dn);
				}
				else
				{
					right_child_offset -= split_nodes_pre_tree;
					left_child_offset -= split_nodes_pre_tree;
					routing_leaf_out[sub2ind(n, s, t, left_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
					= routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;
					routing_leaf_out[sub2ind(n, s, t, right_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
					= routing_split_out[sub2ind(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((double) 1.0 - dn);
				}
			}
		}
	}	


	template <>
	__global__ void kernel_transform<float>(const int num, const int channel,
		const int height, const int width, float* prediction_in, float* prediction_out)
	{
		CUDA_KERNEL_LOOP(index, num * channel * height * width)
		{
			int n = index / (channel * height * width);
			int c = (index / (height * width)) % channel;
			int s = index % (height * width);
			prediction_out[index] = prediction_in[n * height * width * channel + s * channel + c];
		}
	}

	template <>
	__global__ void kernel_transform<double>(const int num, const int channel,
		const int height, const int width, double* prediction_in, double* prediction_out)
	{
		CUDA_KERNEL_LOOP(index, num * channel * height * width)
		{
			int n = index / (channel * height * width);
			int c = (index / (height * width)) % channel;
			int s = index % (height * width);
			prediction_out[index] = prediction_in[n * height * width * channel + s * channel + c];
		}
	}


template <>
__global__ void kernel_updata_all<float>(int num_outer_iter, int num_inner_iter,
          int num_trees, int num_leaf_nodes_per_tree, int num_class, const float* routing_data, 
          const float* class_label_distr_data, float* pred_data) {
  int count = num_outer_iter * num_inner_iter * num_trees * num_class;
  CUDA_KERNEL_LOOP(index, count) {
    int t, k, i, j;
    int idx = index;
    ind2sub(idx, num_inner_iter, num_trees, num_class, &i, &k, &t, &j);
    int pred_idx = sub2ind(i, k, t, j, num_outer_iter, num_inner_iter, num_trees, num_class);
    for(int l = 0; l < num_leaf_nodes_per_tree; l++) {
        int routing_idx = sub2ind(i, k, t, l, num_outer_iter, num_inner_iter, num_trees, num_leaf_nodes_per_tree);
        int distr_idx = sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1);
        pred_data[pred_idx] += routing_data[routing_idx] * class_label_distr_data[distr_idx];
    }
  }
}

template <>
__global__ void kernel_updata_all<double>(int num_outer_iter, int num_inner_iter,
          int num_trees, int num_leaf_nodes_per_tree, int num_class, const double* routing_data, 
          const double* class_label_distr_data, double* pred_data) {
  int count = num_outer_iter * num_inner_iter * num_trees * num_class;
  CUDA_KERNEL_LOOP(index, count) {
    int t, k, i, j;
    int idx = index;
    ind2sub(idx, num_inner_iter, num_trees, num_class, &i, &k, &t, &j);
    int pred_idx = sub2ind(i, k, t, j, num_outer_iter, num_inner_iter, num_trees, num_class);
    for(int l = 0; l < num_leaf_nodes_per_tree; l++) {
        int routing_idx = sub2ind(i, k, t, l, num_outer_iter, num_inner_iter, num_trees, num_leaf_nodes_per_tree);
        int distr_idx = sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1);
        pred_data[pred_idx] += routing_data[routing_idx] * class_label_distr_data[distr_idx];
    }
  }
}

template <> 
__global__ void kernel_update_leaf<float>(int num_trees, int num_leaf_nodes_per_tree, int num_class, int num_outer, int num_inner,
    const float* class_label_distr_data, const float* label_data, const float* routing_leaf_prob_data, const float* tree_prediction_prob_data, 
    float* class_label_distr_temp_data) {
    CUDA_KERNEL_LOOP(index, num_trees * num_leaf_nodes_per_tree * num_class) {
        int t, l, j, i, k;
        int idx = index;
        ind2sub(idx, num_trees, num_leaf_nodes_per_tree, num_class, &i, &t, &l, &j);
        for (i = 0; i < num_outer; i++) {
            for (k = 0; k < num_inner; k++) {
                class_label_distr_temp_data[sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1)] 
                += label_data[sub2ind(i, k, j, 0, num_outer, num_inner, num_class, 1)] 
                * (class_label_distr_data[sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1)] 
                * routing_leaf_prob_data[sub2ind(i, k, t, l, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree)] 
                / fmaxf(tree_prediction_prob_data[sub2ind(i, k, t, j, num_outer, num_inner, num_trees, num_class)], FLT_MIN));
            }
        }
    }
}


template <> 
__global__ void kernel_update_leaf<double>(int num_trees, int num_leaf_nodes_per_tree, int num_class, int num_outer, int num_inner,
    const double* class_label_distr_data, const double* label_data, const double* routing_leaf_prob_data, const double* tree_prediction_prob_data, 
    double* class_label_distr_temp_data) {
    CUDA_KERNEL_LOOP(index, num_trees * num_leaf_nodes_per_tree * num_class) {
        int t, l, j, i, k;
        int idx = index;
        ind2sub(idx, num_trees, num_leaf_nodes_per_tree, num_class, &i, &t, &l, &j);
        for (i = 0; i < num_outer; i++) {
            for (k = 0; k < num_inner; k++) {
                class_label_distr_temp_data[sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1)] 
                += label_data[sub2ind(i, k, j, 0, num_outer, num_inner, num_class, 1)] 
                * (class_label_distr_data[sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1)] 
                * routing_leaf_prob_data[sub2ind(i, k, t, l, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree)] 
                / fmax(tree_prediction_prob_data[sub2ind(i, k, t, j, num_outer, num_inner, num_trees, num_class)], DBL_MIN));
            }
        }
    }
}

/*
template <>
__global__ void kernel_bottom_diff<float>(int num_outer, int num_inner, int num_trees, int num_leaf_nodes_per_tree,
  int num_split_nodes_per_tree, int num_nodes_pre_tree, int num_class, int t, int N, int C, int H, int W,
  const float* class_label_distr_data, const float* routing_leaf_data,
  const float* dn_data, const float* sub_dim_data, const float* tree_prediction_data, 
  const float* label_data, float* inter_var_data, float* bottom_diff) 
{
    CUDA_KERNEL_LOOP(index, num_outer * num_inner) {
        int idx = index;
        int i, k, j, l;
        ind2sub(idx, 1, num_outer, num_inner, &l, &j, &i, &k);
        int pred_idx = sub2ind(i, k, t, j, num_outer, num_inner, num_trees, num_class);
        for (l = 0; l < num_leaf_nodes_per_tree; ++l) {
            for (j = 0; j < num_class; ++j) {
                int inter_idx = sub2ind(i, k, t * num_nodes_pre_tree + num_split_nodes_per_tree + l, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int rout_idx = sub2ind(i, k, t, l, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree);
                int distr_idx = sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1);
                inter_var_data[inter_idx] = class_label_distr_data[distr_idx] * routing_leaf_data[rout_idx] / 
                 max(tree_prediction_data[pred_idx], FLT_MIN);
            }
        }
        for (int n = num_split_nodes_per_tree - 1; n >= 0; n--) {
            int sub_dim_offset = (int) sub_dim_data[sub2ind(t, n, 0, 0, num_trees, num_split_nodes_per_tree, 1, 1)];
            for (int j = 0; j < num_class; ++j) {
                int bottom_idx = sub2ind(i, sub_dim_offset, k/W, k%W, N, C, H, W);
                int inter_idx = sub2ind(i, k, t * num_nodes_pre_tree + n, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int inter_chl_idx = sub2ind(i, k, t * num_nodes_pre_tree + 2 * n + 1, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int inter_chr_idx = sub2ind(i, k, t * num_nodes_pre_tree + 2 * n + 2, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int label_idx = sub2ind(i, j, k/W, k%W, N, num_class, H, W);
                bottom_diff[bottom_idx] += label_data[label_idx] * (dn_data[bottom_idx] * inter_var_data[inter_chr_idx]
                                - ((float)1.0 - dn_data[bottom_idx]) * inter_var_data[inter_chl_idx]);                          
                inter_var_data[inter_idx] = inter_var_data[inter_chl_idx] + inter_var_data[inter_chr_idx];
            }
        }
    }
}


template <>
__global__ void kernel_bottom_diff<double>(int num_outer, int num_inner, int num_trees, int num_leaf_nodes_per_tree,
  int num_split_nodes_per_tree, int num_nodes_pre_tree, int num_class, int t, int N, int C, int H, int W,
  const double* class_label_distr_data, const double* routing_leaf_data,
  const double* dn_data, const double* sub_dim_data, const double* tree_prediction_data, 
  const double* label_data, double* inter_var_data, double* bottom_diff) 
{
    CUDA_KERNEL_LOOP(index, num_outer * num_inner) {
        int idx = index;
        int i, k, j, l;
        ind2sub(idx, 1, num_outer, num_inner, &l, &j, &i, &k);
        int pred_idx = sub2ind(i, k, t, j, num_outer, num_inner, num_trees, num_class);
        for (l = 0; l < num_leaf_nodes_per_tree; ++l) {
            for (j = 0; j < num_class; ++j) {
                int inter_idx = sub2ind(i, k, t * num_nodes_pre_tree + num_split_nodes_per_tree + l, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int rout_idx = sub2ind(i, k, t, l, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree);
                int distr_idx = sub2ind(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1);
                inter_var_data[inter_idx] = class_label_distr_data[distr_idx] * routing_leaf_data[rout_idx] / 
                 max(tree_prediction_data[pred_idx], DBL_MIN);
            }
        }
        for (int n = num_split_nodes_per_tree - 1; n >= 0; n--) {
            int sub_dim_offset = (int) sub_dim_data[sub2ind(t, n, 0, 0, num_trees, num_split_nodes_per_tree, 1, 1)];
            for (int j = 0; j < num_class; ++j) {
                int bottom_idx = sub2ind(i, sub_dim_offset, k/W, k%W, N, C, H, W);
                int inter_idx = sub2ind(i, k, t * num_nodes_pre_tree + n, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int inter_chl_idx = sub2ind(i, k, t * num_nodes_pre_tree + 2 * n + 1, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int inter_chr_idx = sub2ind(i, k, t * num_nodes_pre_tree + 2 * n + 2, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int label_idx = sub2ind(i, j, k/W, k%W, N, num_class, H, W);
                bottom_diff[bottom_idx] += label_data[label_idx] * (dn_data[bottom_idx] * inter_var_data[inter_chr_idx]
                                - ((double)1.0 - dn_data[bottom_idx]) * inter_var_data[inter_chl_idx]);                          
                inter_var_data[inter_idx] = inter_var_data[inter_chl_idx] + inter_var_data[inter_chr_idx];
            }
        }
    }
}
*/

template <>
__global__ void kernel_backward<float>(float* bottom_diff, float* inter_data, const float* cls_lb_distr, const float* label_data, 
                                   const float* routing_lf, const float* dn_data, const float* tree_pred, const float* dim_offset,
                                   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
                                   int h, int w, int num_classes, int tree_id, int num_dims_) {
  int num_nodes = num_split + num_leaf;
  CUDA_KERNEL_LOOP(index, num_outer) 
  {
    for (int i=0; i<num_inner; ++i) 
    {
      for (int l=0; l<num_leaf; ++l) 
      {
        for (int c=0; c<num_classes; ++c) 
        {
          int inter_idx = sub2ind(index,i,tree_id*num_nodes+num_split+l,c, 
                                 num_outer, num_inner, num_trees*num_nodes, num_classes);
          int cls_lb_distr_idx = sub2ind(tree_id, l, c, 0, num_trees, num_leaf, num_classes, 1);
          int routing_lf_idx = sub2ind(index, i, tree_id, l, num_outer, num_inner, num_trees, num_leaf);
          int tree_pred_idx = sub2ind(index, i, tree_id, c, num_outer, num_inner, num_trees, num_classes);
          inter_data[inter_idx] = cls_lb_distr[cls_lb_distr_idx] * routing_lf[routing_lf_idx] / 
                                 fmaxf(tree_pred[tree_pred_idx], FLT_MIN);
        }
      }
      for (int n=num_split-1; n>=0; --n) 
      {
        int dim_offset_idx = sub2ind(tree_id,n,0,0, num_trees,num_split,1,1);
        for (int c=0; c<num_classes; ++c) 
        {
          int lb_idx = sub2ind(index,c,i/w,i%w, num_outer,num_classes,h,w);
          int diff_idx = sub2ind(index,dim_offset[dim_offset_idx],i/w,i%w, num_outer,num_dims_,h,w);
          int inter_left_idx = sub2ind(index,i,tree_id*num_nodes+2*n+1,c,
                                          num_outer,num_inner,num_trees*num_nodes,num_classes);
          int inter_right_idx = inter_left_idx + num_classes;
          const float label_value=label_data[lb_idx];
          bottom_diff[diff_idx] += label_value * (
                    dn_data[diff_idx] * inter_data[inter_right_idx] - 
                    (float(1.0) - dn_data[diff_idx]) * inter_data[inter_left_idx]);
          int inter_parent_idx = sub2ind(index,i,tree_id*num_nodes+n,c,
                                          num_outer,num_inner,num_trees*num_nodes,num_classes);
          inter_data[inter_parent_idx] = inter_data[inter_left_idx] + inter_data[inter_right_idx];
        }
      }
    }
  }
}


template <>
__global__ void kernel_backward<double>(double* bottom_diff, double* inter_data, const double* cls_lb_distr, const double* label_data, 
                                   const double* routing_lf, const double* dn_data, const double* tree_pred, const double* dim_offset,
                                   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
                                   int h, int w, int num_classes, int tree_id, int num_dims_) {
  int num_nodes = num_split + num_leaf;
  CUDA_KERNEL_LOOP(index, num_outer) 
  {
    for (int i=0; i<num_inner; ++i) 
    {
      for (int l=0; l<num_leaf; ++l) 
      {
        for (int c=0; c<num_classes; ++c) 
        {
          int inter_idx = sub2ind(index,i,tree_id*num_nodes+num_split+l,c, 
                                 num_outer, num_inner, num_trees*num_nodes, num_classes);
          int cls_lb_distr_idx = sub2ind(tree_id, l, c, 0, num_trees, num_leaf, num_classes, 1);
          int routing_lf_idx = sub2ind(index, i, tree_id, l, num_outer, num_inner, num_trees, num_leaf);
          int tree_pred_idx = sub2ind(index, i, tree_id, c, num_outer, num_inner, num_trees, num_classes);
          inter_data[inter_idx] = cls_lb_distr[cls_lb_distr_idx] * routing_lf[routing_lf_idx] / 
                                 fmax(tree_pred[tree_pred_idx], DBL_MIN);
        }
      }
      for (int n=num_split-1; n>=0; --n) 
      {
        int dim_offset_idx = sub2ind(tree_id,n,0,0, num_trees,num_split,1,1);
        for (int c=0; c<num_classes; ++c) 
        {
          int lb_idx = sub2ind(index,c,i/w,i%w, num_outer,num_classes,h,w);
          int diff_idx = sub2ind(index,dim_offset[dim_offset_idx],i/w,i%w, num_outer,num_dims_,h,w);
          int inter_left_idx = sub2ind(index,i,tree_id*num_nodes+2*n+1,c,
                                          num_outer,num_inner,num_trees*num_nodes,num_classes);
          int inter_right_idx = inter_left_idx + num_classes;
          const double label_value=label_data[lb_idx];
          bottom_diff[diff_idx] += label_value * (
                    dn_data[diff_idx] * inter_data[inter_right_idx] - 
                    (double(1.0) - dn_data[diff_idx]) * inter_data[inter_left_idx]);
          int inter_parent_idx = sub2ind(index,i,tree_id*num_nodes+n,c,
                                          num_outer,num_inner,num_trees*num_nodes,num_classes);
          inter_data[inter_parent_idx] = inter_data[inter_left_idx] + inter_data[inter_right_idx];
        }
      }
    }
  }
}

template <>
__global__ void kernel_backward_all<float>(float* bottom_diff, float* inter_data, const float* cls_lb_distr, const float* label_data, 
                                   const float* routing_lf, const float* dn_data, const float* tree_pred, const float* dim_offset,
                                   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
                                   int h, int w, int num_classes, int num_dims_) {
  int num_nodes = num_split + num_leaf;
  CUDA_KERNEL_LOOP(index, num_outer) 
  {
    for (int i=0; i<num_inner; ++i) 
    {
      for (int l=0; l<num_leaf; ++l) 
      {
        for(int t= 0; t < num_trees; t++)
        {
          for (int c=0; c<num_classes; ++c) 
          {
            int inter_idx = sub2ind(index,i,t*num_nodes+num_split+l,c, 
                                   num_outer, num_inner, num_trees*num_nodes, num_classes);
            int cls_lb_distr_idx = sub2ind(t, l, c, 0, num_trees, num_leaf, num_classes, 1);
            int routing_lf_idx = sub2ind(index, i, t, l, num_outer, num_inner, num_trees, num_leaf);
            int tree_pred_idx = sub2ind(index, i, t, c, num_outer, num_inner, num_trees, num_classes);
            inter_data[inter_idx] = cls_lb_distr[cls_lb_distr_idx] * routing_lf[routing_lf_idx] / 
                                   fmaxf(tree_pred[tree_pred_idx], FLT_MIN);
          }
      }
    }
      for (int n=num_split-1; n>=0; --n) 
      {
        for(int t = 0; t < num_trees; t++)
        {
          int dim_offset_idx = sub2ind(t,n,0,0, num_trees,num_split,1,1);
          for (int c=0; c<num_classes; ++c) 
          {
            int lb_idx = sub2ind(index,c,i/w,i%w, num_outer,num_classes,h,w);
            int diff_idx = sub2ind(index,dim_offset[dim_offset_idx],i/w,i%w, num_outer,num_dims_,h,w);
            int inter_left_idx = sub2ind(index,i,t*num_nodes+2*n+1,c,
                                            num_outer,num_inner,num_trees*num_nodes,num_classes);
            int inter_right_idx = inter_left_idx + num_classes;
            const float label_value=label_data[lb_idx];
            bottom_diff[diff_idx] += label_value * (
                      dn_data[diff_idx] * inter_data[inter_right_idx] - 
                      (float(1.0) - dn_data[diff_idx]) * inter_data[inter_left_idx]);
            int inter_parent_idx = sub2ind(index,i,t*num_nodes+n,c,
                                            num_outer,num_inner,num_trees*num_nodes,num_classes);
            inter_data[inter_parent_idx] = inter_data[inter_left_idx] + inter_data[inter_right_idx];
          }
        }
      }
    }
  }
}




template <>
__global__ void kernel_backward_all(double* bottom_diff, double* inter_data, const double* cls_lb_distr, const double* label_data, 
                                   const double* routing_lf, const double* dn_data, const double* tree_pred, const double* dim_offset,
                                   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
                                   int h, int w, int num_classes, int num_dims_) {
  int num_nodes = num_split + num_leaf;
  CUDA_KERNEL_LOOP(index, num_outer) 
  {
    for (int i=0; i<num_inner; ++i) 
    {
      for (int l=0; l<num_leaf; ++l) 
      {
        for(int t= 0; t < num_trees; t++)
        {
          for (int c=0; c<num_classes; ++c) 
          {
            int inter_idx = sub2ind(index,i,t*num_nodes+num_split+l,c, 
                                   num_outer, num_inner, num_trees*num_nodes, num_classes);
            int cls_lb_distr_idx = sub2ind(t, l, c, 0, num_trees, num_leaf, num_classes, 1);
            int routing_lf_idx = sub2ind(index, i, t, l, num_outer, num_inner, num_trees, num_leaf);
            int tree_pred_idx = sub2ind(index, i, t, c, num_outer, num_inner, num_trees, num_classes);
            inter_data[inter_idx] = cls_lb_distr[cls_lb_distr_idx] * routing_lf[routing_lf_idx] / 
                                   fmax(tree_pred[tree_pred_idx], DBL_MIN);
          }
      }
    }
      for (int n=num_split-1; n>=0; --n) 
      {
        for(int t = 0; t < num_trees; t++)
        {
          int dim_offset_idx = sub2ind(t,n,0,0, num_trees,num_split,1,1);
          for (int c=0; c<num_classes; ++c) 
          {
            int lb_idx = sub2ind(index,c,i/w,i%w, num_outer,num_classes,h,w);
            int diff_idx = sub2ind(index,dim_offset[dim_offset_idx],i/w,i%w, num_outer,num_dims_,h,w);
            int inter_left_idx = sub2ind(index,i,t*num_nodes+2*n+1,c,
                                            num_outer,num_inner,num_trees*num_nodes,num_classes);
            int inter_right_idx = inter_left_idx + num_classes;
            const double label_value=label_data[lb_idx];
            bottom_diff[diff_idx] += label_value * (
                      dn_data[diff_idx] * inter_data[inter_right_idx] - 
                      (double(1.0) - dn_data[diff_idx]) * inter_data[inter_left_idx]);
            int inter_parent_idx = sub2ind(index,i,t*num_nodes+n,c,
                                            num_outer,num_inner,num_trees*num_nodes,num_classes);
            inter_data[inter_parent_idx] = inter_data[inter_left_idx] + inter_data[inter_right_idx];
          }
        }
      }
    }
  }
}
}

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


#ifndef CAFFE_NEURAL_DECISION_REG_FOREST_LOSS_LAYER_HPP_
#define CAFFE_NEURAL_DECISION_REG_FOREST_LOSS_LAYER_HPP_
#include <vector>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/util/sampling.hpp"

namespace caffe
{
  template <typename Dtype>
  class NeuralDecisionRegForestWithLossLayer : public LossLayer < Dtype >
  {
  public:
    explicit NeuralDecisionRegForestWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
      dn_(new Blob<Dtype>()) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "NeuralDecisionRegForestWithLoss"; }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

     virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
     virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

    void InitRoutingProb();

    void UpdateTreePredictionAllData();

    void UpdateClassLabelDistr();

    void RecordClassLabelDistr();

    void UpdateClassLabelDistrGPU();

    void UpdateTreePredictionAllDataGPU();

    shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;



    int num_trees_;
    int depth_;

    int num_classes_;

    int iter_times_;        //The iterations from the beginning of training.
    int iter_times_in_epoch_;  //How many iterations in an epoch. In each epoch, we update class_label_distr_ first;
    int iter_time_current_epcho_; //The iterations from the beginning of current epoch.
    int iter_time_update_stop_; //The iterations from the time when the class distribution is stopped. 
    //iter_time_current_epcho_ =  iter_times_ % iter_times_in_epoch_.
    int iter_times_class_label_distr_;  //The iterations to converge to a solution for class_label_distr_

    int num_split_nodes_per_tree_;
    int num_leaf_nodes_per_tree_;
    int num_nodes_pre_tree_;

    int num_outer_; // Number of blob N
    int num_inner_; // Number of Pixels in each blob HxW
    int axis_;

    int num_dims_;  // Number of channels of each blob C

    int num_nodes_;

    Dtype scale_;


    /// The random selected tree for the optimization of current mini-batch.
    int tree_for_training_;
    /// bottom vector holder used in call to the underlying SigmoidLayer::Forward (fn)
    vector<Blob<Dtype>*> sigmoid_bottom_vec_;
    /// top vector holder used in call to the underlying SigmoidLayer::Forward (dn)
    vector<Blob<Dtype>*> sigmoid_top_vec_;
    /// the probabilities of sending samples to left subtrees
    shared_ptr<Blob<Dtype> > dn_;
    /// the probabilities that each sample falls into a split node (\mu)
    Blob<Dtype> routing_split_prob_;
    /// the probabilities that each sample falls into a leaf node (\mu)
    Blob<Dtype> routing_leaf_prob_;
    /// the class label distribution of each leaf node
    /// It does not actually hosts the blobs (blobs_ does), so we simply store pointers.
    Blob<Dtype>* mean_;
    Blob<Dtype>* sigma_square_;

    Blob<Dtype>* sub_dimensions_;

    bool b_distr_updated_;
    bool drop_out_;

    int all_data_vec_length_;
    /// the predicted probability density of each sample given by each tree
    Blob<Dtype> tree_prediction_prob_density_;
    /// the prediction of each sample given by each tree
    Blob<Dtype> tree_prediction_;
    /// the intermediate variable used for computing gradient
    Blob<Dtype> inter_var_;

    ///the intermediate variable used for updating leaf node
    vector<shared_ptr<Blob<Dtype> > > routing_leaf_all_data_prob_vec_;

    ///the intermediate variable used for updating leaf node
    vector<shared_ptr<Blob<Dtype> > > tree_prediction_all_data_prob_density_vec_;

    ///the intermediate variable used for updating leaf node
    vector<shared_ptr<Blob<Dtype> > > all_data_label_vec_;

    int num_epoch_;
    /// How to normalize the output loss.
    LossParameter_NormalizationMode normalization_;

    int loss_type_; // 0 for L2 norm loss, 1 for negative log loss
    std::ofstream of_;
    ///the file to store the initailizations of leaf nodes 
    std::ifstream if_;
  };
}
#endif

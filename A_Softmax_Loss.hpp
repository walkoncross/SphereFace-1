#ifndef CAFFE_ANGULARMARGIN_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_ANGULARMARGIN_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "AngularMargin fully-connected" layer, computes an
 * AngularMargin inner product with a set of normalized  weights, and zero biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class AngularMarginInnerProductLayer : public Layer<Dtype> 
{
 public:
  explicit AngularMarginInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AngularMarginInnerProduct"; }

  virtual inline int ExactNumBottomBlobs() const { return 2;
  }//输入需要多少个blob

  virtual inline int MaxTopBlobs() const { return 2; }//输出最多多少个blob

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;//batch图片数目
  int K_;//特征维度
  int N_;//输出分类数目
  

  //modify below 
  AngularMarginInnerProductParameter_AngularMarginType type_;//按照angular
  //margin 角度间隔  分类，m

  // common variables
  Blob<Dtype> x_norm_;//输入特征的范数
  Blob<Dtype> w_norm_;//权重的范数
  Blob<Dtype> w_norm_scalar;
  Blob<Dtype> cos_theta_;//余弦角度
  Blob<Dtype> sign_0_; // sign_0 = sign(cos_theta)

  // for DOUBLE type
  Blob<Dtype> cos_theta_quadratic_;//pow(cos_theta_,2);

  // for TRIPLE type
  Blob<Dtype> sign_1_; // sign_1 = sign(abs(cos_theta) - 0.5)
  Blob<Dtype> sign_2_; // sign_2 = sign_0 * (1 + sign_1) - 2
  Blob<Dtype> cos_theta_cubic_;

  // for QUADRA type
  Blob<Dtype> sign_3_; // sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
  Blob<Dtype> sign_4_; // sign_4 = 2 * sign_0 + sign_3 - 3
  Blob<Dtype> cos_theta_quartic_;

  int iter_;//根据前向传播次数，不断调整lamda
  Dtype lambda_;//计算fyi时使用的加权系数

};

}  // namespace caffe

#endif  // CAFFE_ANGULARMARGIN_INNER_PRODUCT_LAYER_HPP_

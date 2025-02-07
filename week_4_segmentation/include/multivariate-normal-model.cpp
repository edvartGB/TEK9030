#include "multivariate-normal-model.h"
#include "opencv2/imgproc.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>

MultivariateNormalModel::MultivariateNormalModel(const cv::Mat &samples) {
  performTraining(samples);
}

void MultivariateNormalModel::performTraining(const cv::Mat &samples) {
  // samples.shape: num_samples x 3 (RGB)
  cv::reduce(samples, mean_, 0, cv::REDUCE_AVG);
  cv::calcCovarMatrix(samples, covariance_, mean_,
                      cv::COVAR_NORMAL | cv::COVAR_ROWS);
  covariance_ = covariance_ * (1.0 / (samples.size[0] - 1.0));

  if (cv::abs(cv::determinant(covariance_)) < 1e-14) {
    covariance_ += cv::Mat::eye(covariance_.size(), CV_32F) * 1e-6;
  }

  inverse_covariance_ = covariance_.inv();
}

cv::Mat MultivariateNormalModel::computeMahalanobisDistances(
    const cv::Mat &image) const {
  cv::Mat float_image;
  image.convertTo(float_image, CV_64F);
  const auto num_samples = float_image.total();
  float_image = float_image.reshape(1, static_cast<int>(num_samples));

  cv::Mat mahalanobis_img(image.size(), CV_32F);

  // TODO 2: Compute the mahalanobis distance for each pixel feature vector wrt
  // the multivariate normal model.
  mahalanobis_img.setTo(std::numeric_limits<float>::infinity());

  for (int i = 0; i < num_samples; ++i) {
    cv::Mat x(1, float_image.cols, CV_64F, float_image.ptr<float>(i));
    double d = cv::Mahalanobis(x, mean_, inverse_covariance_);

    int row = i / image.cols;  
    int col = i % image.cols;  
    mahalanobis_img.at<float>(row, col) = static_cast<float>(d);
  }

  return mahalanobis_img;
}

cv::Mat MultivariateNormalModel::mean() const { return mean_.clone(); }

cv::Mat MultivariateNormalModel::covariance() const {
  return covariance_.clone();
}

cv::Mat MultivariateNormalModel::inverseCovariance() const {
  return inverse_covariance_.clone();
}
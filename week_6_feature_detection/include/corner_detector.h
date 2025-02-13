#pragma once

#include "opencv2/opencv.hpp"
#include "filters.h"

/// \brief Valid corner metrics.
enum class CornerMetric
{
  harris, 
  harmonic_mean, 
  min_eigen
};


/// \brief A home brewed corner detector!
class CornerDetector
{
public:
  /// \brief Constructs the corner detector.
  /// \param metric The metric used to extract corners.
  /// \param do_visualize Shows additional debug/educational figures when true.
  /// \param quality_level The quality level used for thresholding areas with corners.
  /// \param gradient_sigma The standard deviation for the gradient filter.
  /// \param window_sigma The standard deviation for the window filters.
  explicit CornerDetector(CornerMetric metric = CornerMetric::harris,
                          bool do_visualize = false,
                          float quality_level = 0.01f,
                          float gradient_sigma = 1.0f,
                          float window_sigma = 2.0f);

  /// \brief Detects corners in an image.
  /// \param image The image that is queried for corners.
  /// \return An array of corner key points.
  std::vector<cv::KeyPoint> detect(const cv::Mat& image) const;

private:
  cv::Mat harrisMetric(const cv::Mat& A, const cv::Mat& B, const cv::Mat& C) const;

  cv::Mat harmonicMeanMetric(const cv::Mat& A, const cv::Mat& B, const cv::Mat& C) const;

  cv::Mat minEigenMetric(const cv::Mat& A, const cv::Mat& B, const cv::Mat& C) const;

  CornerMetric metric_type_;
  bool do_visualize_;
  float quality_level_;
  float window_sigma_;
  cv::Mat g_kernel_;
  cv::Mat dg_kernel_;
  cv::Mat win_kernel_;
};

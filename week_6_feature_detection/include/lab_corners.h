#pragma once

#include "circle_estimator.h"
#include "opencv2/core.hpp"

/// \brief Runs the lab.
void runLabCorners();

/// \brief Converts OpenCV keypoints to an Eigen matrix of Vector2f columns.
/// \param keypoints Vector of keypoints.
/// \return Matrix where each columns is the corresponding point.
Eigen::Matrix2Xf convertToPoints(const std::vector<cv::KeyPoint>& keypoints);

/// \brief Draws the corner detection results.
/// \param img The image that will be drawn on.
/// \param keypoints The corner keypoints.
/// \param time The processing duration for corner detection.
void drawCornerResult(const cv::Mat& img,
                      const std::vector<cv::KeyPoint>& keypoints,
                      double time);

/// \brief Draws the circle estimation results.
/// \param img The image that will be drawn on.
/// \param keypoints The corner keypoints.
/// \param estimate The circle estimation result.
/// \param time The processing duration for circle estimation.
void drawCircleResult(const cv::Mat& img,
                      const std::vector<cv::KeyPoint>& keypoints,
                      const CircleEstimate& estimate,
                      double time);

#pragma once

#include "circle.h"

// Make shorthand alias for logical vector.
using LogicalVector = Eigen::Matrix<bool, 1, Eigen::Dynamic>;

/// \brief Datatype for circle estimate as a result from CircleEstimator.
struct CircleEstimate
{
  Circle circle;
  int num_iterations;
  Eigen::Index num_inliers;
  LogicalVector is_inlier;
};

/// \brief A robust circle estimator based on circle point measurements.
class CircleEstimator
{
public:
  /// \brief Constructs a circle estimator.
  /// \param p The desired probability of getting a good sample.
  /// \param distance_threshold The maximum distance a good sample can have from the circle.
  explicit CircleEstimator(double p = 0.99, float distance_threshold = 5.0f);

  /// \brief Estimates a circle based on the point measurements using RANSAC.
  /// \param points Point measurements on the circle corrupted by noise.
  /// \return The circle estimate based on the entire inlier set.
  CircleEstimate estimate(const Eigen::Matrix2Xf& points) const;

private:

  CircleEstimate ransacEstimator(const Eigen::Matrix2Xf& pts) const;

  Circle leastSquaresEstimator(const Eigen::Matrix2Xf& pts) const;

  Eigen::Matrix2Xf extractInlierPoints(const CircleEstimate& estimate, const Eigen::Matrix2Xf& pts) const;

  double p_;
  float distance_threshold_;
};

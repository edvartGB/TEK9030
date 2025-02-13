#include "circle_estimator.h"
#include <random>

CircleEstimator::CircleEstimator(double p, float distance_threshold)
    : p_{p}
    , distance_threshold_{distance_threshold}
{}

CircleEstimate CircleEstimator::estimate(const Eigen::Matrix2Xf& points) const
{
  if (points.cols() < 3)
  {
    // Too few points to estimate any circle.
    return {};
  }

  // Estimate circle using RANSAC.
  CircleEstimate estimate = ransacEstimator(points);
  if (estimate.num_inliers == 0)
  { return {}; }

  // Extract the inlier points.
  Eigen::Matrix2Xf inlier_pts = extractInlierPoints(estimate, points);

  // Estimate circle based on all the inliers.
  estimate.circle = leastSquaresEstimator(inlier_pts);

  return estimate;
}

CircleEstimate CircleEstimator::ransacEstimator(const Eigen::Matrix2Xf& pts) const
{
  // Initialize best set.
  Eigen::Index best_num_inliers{0};
  Circle best_circle;
  LogicalVector best_is_inlier;

  // Set up random number generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> uni_dist(0, static_cast<int>(pts.cols()-1));

  // Initialize maximum number of iterations.
  int max_iterations = std::numeric_limits<int>::max();

  // Perform RANSAC.
  int iterations{0};
  for (; iterations < max_iterations; ++iterations)
  {
    // Determine test circle by drawing minimal number of samples.
    Circle tst_circle(pts.col(uni_dist(gen)),
                      pts.col(uni_dist(gen)),
                      pts.col(uni_dist(gen)));

    // Count number of inliers.
    LogicalVector is_inlier = tst_circle.distance(pts).array() < distance_threshold_;
    Eigen::Index tst_num_inliers = is_inlier.count();

    // Check if this estimate gave a better result.
    // Todo 8: Remove break and perform the correct test!
    break;     // Remove!
    if (false) // Perform the correct test!
    {
      // Update circle with largest inlier set.
      best_circle = tst_circle;
      best_num_inliers = tst_num_inliers;
      best_is_inlier = is_inlier;

      // Update max iterations.
      double inlier_ratio = static_cast<double>(best_num_inliers) / static_cast<double>(pts.cols());
      max_iterations = static_cast<int>(std::log(1.0 - p_) / std::log(1.0 - inlier_ratio*inlier_ratio*inlier_ratio));
    }
  }

  return {best_circle, iterations, best_num_inliers, best_is_inlier};
}

Circle CircleEstimator::leastSquaresEstimator(const Eigen::Matrix2Xf& pts) const
{
  // The equations for the points (x_i, y_i) on the circle (x_c, y_c, r) is:
  //     (x_i - x_c)^2 + (y_i - y_c)^2 = r^2
  //
  // By multiplying out, we get the linear equations
  //     (2*x_c)*x_i + (2*y_c)*y_i + (r^2 - x_c^2 - y_x^2) = x_i^2 + y_i^2
  //
  // The least-squares problem then has the form A*p = b, where
  //     A = [x_i, y_i, 1],
  //     p = [2*x_c, 2*y_c, r^2 - x_c^2 - y_x^2]^T,
  //     b = [x_i^2 + y_i^2]
  //
  // by solving for p = [p_0, p_1, p_2], we get the following estimates for the circle parameters:
  //     x_c = 0.5 * p_0,
  //     y_c = 0.5 * p_1,
  //     r = sqrt(p_2 + x_c^2 + y_c^2)

  // Construct A and b.
  Eigen::MatrixXf A(pts.cols(), 3);
  A.leftCols(2) = pts.transpose();
  A.col(2).setConstant(1.0f);
  Eigen::VectorXf b = pts.colwise().squaredNorm();

  // Determine solution for p.
  // See https://eigen.tuxfamily.org/dox-devel/group__LeastSquares.html
  Eigen::Vector3f p = A.colPivHouseholderQr().solve(b);

  // Extract center point and radius from the parameter vector p.
  Eigen::Vector2f center_point = 0.5f * p.head<2>();
  float radius = std::sqrt(p(2) + center_point.squaredNorm());

  return {center_point, radius};
}

Eigen::Matrix2Xf CircleEstimator::extractInlierPoints(const CircleEstimate& estimate, const Eigen::Matrix2Xf& pts) const
{
  Eigen::Matrix2Xf inliers(2, estimate.num_inliers);

  int curr_col = 0;
  for (int i = 0; i < pts.cols(); ++i)
  {
    if (estimate.is_inlier(i))
    {
      inliers.col(curr_col++) = pts.col(i);
    }
  }

  return inliers;
}

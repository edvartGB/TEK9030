#include "lab_corners.h"
#include "corner_detector.h"
#include <chrono>

// Make shorthand aliases for timing tools.
using Clock = std::chrono::high_resolution_clock;
using DurationInMs = std::chrono::duration<double, std::milli>;

void runLabCorners()
{
  // Open video stream from camera.
  const int camera_id = 0; // Should be 0 or 1 on the lab PCs.
  cv::VideoCapture cap(camera_id);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  if (!cap.isOpened())
  {
    throw std::runtime_error("Could not open camera by index " + std::to_string(camera_id));
  }

  // Create window.
  const std::string win_name = "Lab: Estimating circles from corners";
  cv::namedWindow(win_name);

  // Construct the corner detector.
  // Play around with the parameters!
  // When the second argument is true, additional debug visualizations are shown.
  CornerDetector det(CornerMetric::harmonic_mean, true, 0.01f, 0.85f, 2.0f);

  // Construct the circle estimator.
  CircleEstimator estimator;
  while (true)
  {
    // Read a frame from the camera.
    cv::Mat frame;
    cap >> frame;

    if (frame.empty())
    { break; }

    // Convert frame to gray scale image.
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

    // Perform corner detection.
    // Measure how long the processing takes.
    auto start = Clock::now();
    std::vector<cv::KeyPoint> corners = det.detect(gray_frame);
    auto end = Clock::now();
    DurationInMs corner_proc_duration = end - start;

    // Keep the highest scoring points.
    const int num_to_keep = 1000;
    cv::KeyPointsFilter::retainBest(corners, num_to_keep);

    // Convert corners to Eigen points.
    Eigen::Matrix2Xf points = convertToPoints(corners);

    // Estimate circle from points.
    // Measure how long the processing takes.
    start = Clock::now();
    CircleEstimate estimate = estimator.estimate(points);
    end = Clock::now();
    DurationInMs circle_proc_duration = end - start;

    // Visualize the results.
    cv::Mat vis_img = frame.clone();
    drawCornerResult(vis_img, corners, corner_proc_duration.count());
    drawCircleResult(vis_img, corners, estimate, circle_proc_duration.count());
    cv::imshow(win_name, vis_img);
    if (cv::waitKey(30) >= 0) break;
  }
}

Eigen::Matrix2Xf convertToPoints(const std::vector<cv::KeyPoint>& keypoints)
{
  // Convert each OpenCV point to column vector in Eigen matrix.
  Eigen::Matrix2Xf points(2, keypoints.size());
  for (size_t i=0; i < keypoints.size(); ++i)
  {
    points.col(i) = Eigen::Vector2f(keypoints[i].pt.x, keypoints[i].pt.y);
  }

  return points;
}

void drawCornerResult(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, double time)
{
  // Print processing duration.
  std::stringstream duration_info;
  duration_info << std::fixed << std::setprecision(0); // Set to 0 decimals.
  duration_info << "Corner time: " << time << "ms";
  cv::putText(img, duration_info.str(), {10, 20}, cv::FONT_HERSHEY_PLAIN, 1.0, {0, 255, 0});

  // Draw corners.
  cv::drawKeypoints(img, keypoints, img, cv::Scalar{0,255,0});
}

void drawCircleResult(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const CircleEstimate& estimate,
                      double time)
{
  // Check if there is a result.
  if (estimate.num_inliers == 0)
  {
    return;
  }

  // Print processing duration.
  std::stringstream duration_info;
  duration_info << std::fixed << std::setprecision(0); // Set to 0 decimals.
  duration_info << "Circle time: " << time << "ms";
  cv::putText(img, duration_info.str(), {10, 40}, cv::FONT_HERSHEY_PLAIN, 1.0, {0, 0, 255});

  // Extract and draw circle point inliers.
  std::vector<cv::KeyPoint> inlier_keypts;
  for (size_t i=0; i<keypoints.size(); ++i)
  {
    if (estimate.is_inlier(i))
    {
      inlier_keypts.push_back(keypoints[i]);
    }
  }
  cv::drawKeypoints(img, inlier_keypts, img, cv::Scalar{0,0,255});


  // Draw estimated circle
  const Eigen::Vector2i center = estimate.circle.center().array().round().cast<int>();
  cv::Point center_point{center.x(), center.y()};
  int radius = static_cast<int>(std::round(estimate.circle.radius()));
  cv::circle(img, center_point, radius, cv::Scalar(0, 0, 255), cv::LINE_4, cv::LINE_AA);

  // Print some information about the estimation.
  std::stringstream iterations_info;
  iterations_info << "Iterations: " << estimate.num_iterations;
  cv::putText(img, iterations_info.str(), {10, 60}, cv::FONT_HERSHEY_PLAIN, 1.0, {0, 0, 255});

  std::stringstream inliers_info;
  inliers_info << "Inliers: " << estimate.num_inliers;
  cv::putText(img, inliers_info.str(), {10, 80}, cv::FONT_HERSHEY_PLAIN, 1.0, {0, 0, 255});
}

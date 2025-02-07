#include <opencv2/imgcodecs.hpp>
#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
#include "../third_party/Dense"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include "Opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>

#define PI 3.14159
#define RAD2DEG (180.0f/PI)
#define DEG2RAD (PI/180.0f)

int main() {
  cv::Mat img = cv::imread("img_grid.png");
  std::string window_title{"Lab 1.2: Original image"};
  cv::namedWindow(window_title, cv::WINDOW_NORMAL);

  float theta = 45.0f*DEG2RAD;
  Eigen::Matrix2f R {
    {cos(theta), -sin(theta)},
    {sin(theta), cos(theta)},
  };
  Eigen::Matrix<float, 3, 3> E_rot;
  E_rot << R, Eigen::Vector2f(0.0, 0.0),
       0, 0, 1;

  Eigen::Matrix<float, 3, 3> E_center_to_origin;
  E_center_to_origin << 1, 0,  -img.cols/2.0f,
       0, 1,  -img.rows/2.0f,
       0, 0, 1;
  Eigen::Matrix<float, 3, 3> E_origin_to_center = E_center_to_origin.inverse();
  
  Eigen::Matrix<float, 3, 3> S;
  S << 0.3, 0, 0,
      0, 0.3, 0,
      0, 0, 1;

  Eigen::Matrix<float, 3, 3> E_composed;
  E_composed = E_origin_to_center*S*E_rot*E_center_to_origin;
  
  cv::Mat RotCentre_cv;
  cv::eigen2cv(E_composed, RotCentre_cv);

  cv::Mat warpedImg;
  cv::warpPerspective(img, warpedImg, RotCentre_cv, img.size(), cv::INTER_CUBIC);


  while (true){
    cv::imshow(window_title, warpedImg);
     if (cv::waitKey(10.0) >= 0) {
      break;
    } 
  }
  return 0;
}
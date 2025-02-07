#include <opencv2/core/hal/interface.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
#include "../third_party/Dense"
#include "Opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#define PYR_SIZE 10
#define PYR_MAX_IDX PYR_SIZE - 1

cv::Mat getMask(const int nRows, const int nCols) {
  cv::Mat mask = cv::Mat::zeros(nRows, nCols, CV_32FC3);
  int maskRadius = pow(mask.cols/4, 2.0);
  int maskCx = mask.rows/4;
  int maskCy = mask.cols/4;
  for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
      for (int k = 0; k < mask.channels(); k++) {
        int R = pow(j-maskCx, 2.0) + pow(i-maskCy, 2.0);
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        // if (R < maskRadius) {
          // mask.at<cv::Vec3f>(i, j) = cv::Vec3f(1.0f, 1.0f, 1.0f);
        if (r < 0.5){
          mask.at<cv::Vec3f>(i, j) = cv::Vec3f(r, 0.0, 1.0);
        } else {
          mask.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 0.0f);
        }
      }
    }
  }
  cv::blur(mask, mask, cv::Size(1, 1));
  return mask;
}

cv::Mat linearBlending(const cv::Mat &img_1, const cv::Mat &img_2,
                       const cv::Mat &weights) {
  cv::Mat result;
  result = img_1.mul(weights) + img_2.mul(cv::Scalar(1.0, 1.0, 1.0) - weights);
  return result;
}

std::vector<cv::Mat> constructGaussianPyramid(const cv::Mat &img) {
  std::vector<cv::Mat> gaussianPyramid;
  gaussianPyramid.push_back(img);
  for (int i = 0; i < PYR_SIZE - 1; i++) {
    cv::Mat down;
    cv::pyrDown(gaussianPyramid[i], down);
    gaussianPyramid.push_back(down);
  }
  return gaussianPyramid;
}

std::vector<cv::Mat> constructLaplacianPyramid(const cv::Mat &img) {
  std::vector<cv::Mat> gaussianPyramid = constructGaussianPyramid(img);
  std::vector<cv::Mat> laplacianPyramid;

  laplacianPyramid.push_back(gaussianPyramid[gaussianPyramid.size() - 1]);
  for (int i = 0; i < PYR_SIZE - 1; i++) {
    cv::Mat upScaled;
    cv::pyrUp(gaussianPyramid[PYR_MAX_IDX - i], upScaled);
    cv::resize(upScaled, upScaled, gaussianPyramid[PYR_MAX_IDX - i - 1].size());
    laplacianPyramid.push_back(gaussianPyramid[PYR_MAX_IDX - i - 1] - upScaled);
  }
  std::reverse(laplacianPyramid.begin(), laplacianPyramid.end());
  return laplacianPyramid;
}

cv::Mat collapsePyramid(const std::vector<cv::Mat> &pyr) {
  for (int i = pyr.size() - 1; i >= 1; i--) {
    cv::Mat upScaled;
    cv::pyrUp(pyr[i], upScaled);
    cv::resize(upScaled, upScaled, pyr[i - 1].size());
    pyr[i - 1] += upScaled;
  }
  return pyr[0];
}

cv::Mat laplaceBlending(const cv::Mat &img_1, const cv::Mat &img_2,
                        const cv::Mat &weights) {
  std::vector<cv::Mat> weights_pyr = constructGaussianPyramid(weights);
  std::vector<cv::Mat> img1_pyr = constructLaplacianPyramid(img_1);
  std::vector<cv::Mat> img2_pyr = constructLaplacianPyramid(img_2);

  std::vector<cv::Mat> blended_pyr;
  for (int i = 0; i < weights_pyr.size(); i++) {
    blended_pyr.push_back(
        linearBlending(img1_pyr[i], img2_pyr[i], weights_pyr[i]));
  }
  return collapsePyramid(blended_pyr);
}

int main() {
  cv::Mat img1 = cv::imread("white_tiger.png");
  cv::Mat img2 = cv::imread("tiger.png");
  std::string window_title{"Lab 1.2: Original image"};
  cv::namedWindow(window_title, cv::WINDOW_NORMAL);

  img1.convertTo(img1, CV_32F, 1.0 / 255.0);
  img2.convertTo(img2, CV_32F, 1.0 / 255.0);
  cv::Mat mask = getMask(img1.rows, img1.cols);
  cv::Mat linearBlend = linearBlending(img1, img2, mask);

  std::vector<cv::Mat> img1GPyramid = constructGaussianPyramid(img1);
  std::vector<cv::Mat> img2GPyramid = constructGaussianPyramid(img2);
  std::vector<cv::Mat> img1LPyramid = constructLaplacianPyramid(img1);
  std::vector<cv::Mat> img2LPyramid = constructLaplacianPyramid(img2);
  std::vector<cv::Mat> blends;
  std::vector<cv::Mat> masks;
  blends.push_back(linearBlend);
  blends.push_back(laplaceBlending(img1, img2, mask));
  masks.push_back(mask);

  int i = 0;
  int keyCode = -1;
  std::vector<cv::Mat> displayImgs = blends;
  while (true) {
    keyCode = cv::waitKey(10.0);
    switch (keyCode) {
    case 49:
      displayImgs = img1GPyramid;
      break;
    case 50:
      displayImgs = img2GPyramid;
      break;
    case 51:
      displayImgs = img1LPyramid;
      break;
    case 52:
      displayImgs = img2LPyramid;
      break;
    case 53:
      displayImgs = blends;
      break;
    case 54:
      displayImgs = masks;
      break;
    case 13:
      i++;
      break;
    default:
      break;
    }
    i = i % displayImgs.size();
    cv::imshow(window_title, displayImgs[i]);
  }
  return 0;
}
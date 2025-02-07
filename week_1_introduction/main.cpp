#include "Opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>

int main() {
  constexpr int device_id = 1;

  cv::VideoCapture input_stream(device_id);
  if (!input_stream.isOpened()) {
    std::cerr << "failed to open stream" << std::endl;
    return 1;
  }
  const std::string window_title = "LAB_0";
  cv::namedWindow(window_title, cv::WINDOW_GUI_NORMAL);

  cv::Mat frame;
  cv::Mat prevFrame;
  cv::Mat result;
  constexpr int delay_ms = 15;

  input_stream >> prevFrame;
  while (true) {
    input_stream >> frame;
    if (frame.empty()) {
      std::cerr << "camera returned empty frame" << std::endl;
      return 1;
    }

    cv::absdiff(frame, prevFrame, result);
    result = cv::abs(result);
    cv::imshow(window_title, result);
    if (cv::waitKey(delay_ms) >= 0) {
      break;
    }
    prevFrame = frame.clone();
  }
  return 0;
}
#include "filters.h"

cv::Mat create1DGaussianKernel(float sigma, int radius)
{
  if (radius <= 0) radius = static_cast<int>(std::ceil(3.5f * sigma));

  const int length = 2*radius + 1;
  cv::Mat kernel(length, 1, CV_32F);

  const float factor = -0.5f / (sigma*sigma);
  float sum = 0.f;

  for (int i = 0; i < length; ++i)
  {
    const auto x = static_cast<float>(i - radius);
    const float kernel_element = std::exp(x * x * factor);
    kernel.at<float>(i) = kernel_element;
    sum += kernel_element;
  }

  kernel /= sum;
  return kernel;
}

cv::Mat create1DDerivatedGaussianKernel(float sigma, int radius)
{
  if (radius <= 0) radius = static_cast<int>(std::ceil(3.5f * sigma));

  const int length = 2*radius+1;
  cv::Mat kernel(2*radius+1, 1, CV_32F);
  cv::Mat gKernel = create1DGaussianKernel(sigma, radius);

  float sum = 0.0f;
  for (int i = 0; i < length;i++){
    const auto x = static_cast<float>(i-radius);
    const float kernel_element = -x*(sigma*sigma)*gKernel.at<float>(i);
    kernel.at<float>(i) = kernel_element;
    sum += kernel_element;
  }
  if (sum > 0.0000001){
    kernel /= sum; 
  }
  return kernel;
}

#include <opencv2/opencv.hpp>

namespace fdf {
  void applyFilter(const cv::Mat &src, const cv::Mat &mask, cv::Mat &dst);
  cv::Mat idealLpf(const cv::Mat &src, double d0);
  cv::Mat getILpfKernel(double d0, int width, int height);
  cv::Mat gaussianLpf(const cv::Mat &src, double d0);
  cv::Mat getGLpfKernel(double d0, int width, int height);
}
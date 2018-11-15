#include <opencv2/opencv.hpp>

namespace fdf {
  void applyFilter(const cv::Mat &src, const cv::Mat &mask, cv::Mat &dst);
  cv::Mat idealLpf(const cv::Mat &src, double d0);
  cv::Mat getILpfKernel(double d0, int width, int height);
  cv::Mat gaussianLpf(const cv::Mat &src, double d0);
  cv::Mat getGLpfKernel(double d0, int width, int height);
  cv::Mat ButterworthLpf(const cv::Mat &src, double d0, int order);
  cv::Mat getBLpfKernel(double d0, int order, int width, int height);
  cv::Mat HomomorphicLpf(const cv::Mat &src, double gamma_l, 
                         double gamma_h, double c, double d0);
  cv::Mat getHLpfKernel( double gamma_l, double gamma_h, double c, 
                         double d0, int width, int height);
}
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <thread>
#include <opencv2/opencv.hpp>

class Dft2d {
  public:
  Dft2d(cv::Mat &src_img);
  cv::Mat getDftImg();
  private:
  cv::Mat src_img_;
  std::complex<double> getDftValue(int u, int v);
};

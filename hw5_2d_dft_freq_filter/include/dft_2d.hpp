#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <opencv2/opencv.hpp>

class Dft2d {
  public:
  Dft2d(cv::Mat &src_img);
  cv::Mat getDftImg();
  cv::Mat getDftImg(int num_threads);
  private:
  cv::Mat src_img_;
  std::complex<double> getDftValue(int u, int v);
  int dtf_p_count_;
  std::mutex dtf_p_count_mtx_; 
  void dftTask(int min_rows, int max_rows, cv::Mat &dft_out);
  void dftProgress();
};

class IDft2d {
  public:
  IDft2d(cv::Mat &src_img);
  cv::Mat getIDftImg();
  private:
  cv::Mat src_img_;
  std::complex<double> getDftValue(int u, int v);
};
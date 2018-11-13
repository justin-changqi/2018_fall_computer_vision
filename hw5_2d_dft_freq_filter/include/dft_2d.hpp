#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "histogram_equalization.hpp"

class Dft2d {
  public:
  Dft2d(cv::Mat &src_img);
  void computeDft(int num_threads);
  cv::Mat getSpectrumImg();
  cv::Mat getRealMat();
  cv::Mat getImagMat();
  private:
  cv::Mat src_img_;
  cv::Mat Re_;
  cv::Mat Im_;
  std::complex<double> getDftValue(int u, int v);
  int dtf_p_count_;
  std::mutex dtf_p_count_mtx_; 
  void dftTask(int min_rows, int max_rows);
  void dftProgress();
};

class IDft2d {
  public:
  IDft2d(const cv::Mat &re, const cv::Mat &im);
  void computeIDft(int num_threads);
  cv::Mat getInvImg();

  private:
  cv::Mat inv_img_;
  cv::Mat Re_;
  cv::Mat Im_;
  double getIDftValue(int x, int y);
  int dtf_p_count_;
  std::mutex dtf_p_count_mtx_; 
  void IdftTask(int min_rows, int max_rows);
  void IdftProgress();
};
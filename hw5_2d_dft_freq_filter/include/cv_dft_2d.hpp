#include <iostream>
#include <opencv2/opencv.hpp>

class CvDft2d {
  public:
  CvDft2d(cv::Mat &src_img);
  void computeDft();
  cv::Mat getSpectrumImg();
  static cv::Mat getSpectrumImg(const cv::Mat &complex);
  cv::Mat getRealMat();
  cv::Mat getImagMat();
  cv::Mat getComplex();
  private:
  cv::Mat src_img_;
  cv::Mat complex_;
};

class CvIDft2d {
  public:
  CvIDft2d(const cv::Mat &re, const cv::Mat &im);
  CvIDft2d(const cv::Mat &complex);
  void computeIDft();
  cv::Mat getInvImg();

  private:
  cv::Mat inv_img_;
  cv::Mat complex_;
};

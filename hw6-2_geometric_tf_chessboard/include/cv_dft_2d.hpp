#include <iostream>
#include <opencv2/opencv.hpp>

class CvDft2d {
  public:
  CvDft2d(cv::Mat &src_img);
  void computeDft();
  cv::Mat getSpectrumImg();
  cv::Mat getRealMat();
  cv::Mat getImagMat();
  private:
  cv::Mat src_img_;
  cv::Mat complex_;
};

class CvIDft2d {
  public:
  CvIDft2d(const cv::Mat &re, const cv::Mat &im);
  void computeIDft();
  cv::Mat getInvImg();

  private:
  cv::Mat inv_img_;
  cv::Mat complex_;
};

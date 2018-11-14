#include <cv_dft_2d.hpp>

// ref: Opencv DFT
// https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html

CvDft2d::CvDft2d(cv::Mat &src_img)
{
  int m = cv::getOptimalDFTSize( src_img.rows );
  int n = cv::getOptimalDFTSize( src_img.cols );
  cv::copyMakeBorder(src_img, src_img_, 0, m - src_img.rows, 0, n - src_img.cols, 
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));
  cv::Mat planes[] = {cv::Mat_<float>(src_img_), cv::Mat::zeros(src_img_.size(), CV_32F)};
  cv::merge(planes, 2, complex_);
  this->computeDft();
}

void CvDft2d::computeDft()
{
  cv::dft(complex_, complex_);
}

cv::Mat CvDft2d::getSpectrumImg()
{
  cv::Mat planes[] = {cv::Mat::zeros(src_img_.size(), CV_32F), cv::Mat::zeros(src_img_.size(), CV_32F)};
  cv::split(complex_, planes);
  cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
  cv::Mat magI = planes[0];
  magI += cv::Scalar::all(1);                    // switch to logarithmic scale
  log(magI, magI);
  magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
  int cx = magI.cols/2;
  int cy = magI.rows/2;
  cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
  cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
  cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
  cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);
  cv::normalize(magI, magI, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  return magI.clone();
}

cv::Mat CvDft2d::getRealMat()
{
  cv::Mat planes[] = {cv::Mat_<float>(src_img_), cv::Mat::zeros(src_img_.size(), CV_32F)};
  cv::split(complex_, planes);
  return planes[0];
}

cv::Mat CvDft2d::getImagMat()
{
  cv::Mat planes[] = {cv::Mat_<float>(src_img_), cv::Mat::zeros(src_img_.size(), CV_32F)};
  cv::split(complex_, planes);
  return planes[1];
}

cv::Mat CvDft2d::getComplex()
{
  return complex_.clone();
}

CvIDft2d::CvIDft2d(const cv::Mat &re, const cv::Mat &im)
{
  cv::Mat planes[] = {re, im};
  cv::merge(planes, 2, complex_);
}

CvIDft2d::CvIDft2d(const cv::Mat &complex)
{
  complex_ = complex.clone();
  this->computeIDft();
}

void CvIDft2d::computeIDft()
{
  cv::dft(complex_, inv_img_, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
}

cv::Mat CvIDft2d::getInvImg()
{
  cv::normalize(inv_img_, inv_img_, 0, 255, CV_MINMAX, CV_8UC1);
  return inv_img_.clone();
}



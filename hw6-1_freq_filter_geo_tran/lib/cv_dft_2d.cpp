#include <cv_dft_2d.hpp>

// ref: Opencv DFT
// https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html

CvDft2d::CvDft2d(cv::Mat &src_img)
{
  src_img.convertTo(src_img_, CV_32FC1);
  std::vector<cv::Mat> channels;
  channels.push_back(src_img_);
  channels.push_back(src_img_);
  merge(channels, complex_);
  this->computeDft();
}

void CvDft2d::computeDft()
{
  cv::dft(complex_, complex_, cv::DFT_COMPLEX_OUTPUT);
}

cv::Mat CvDft2d::getSpectrumImg()
{
  std::vector<cv::Mat> planes(2);
  cv::split(complex_, planes);
  cv::Mat img_out(complex_.size(), CV_32FC1);
  cv::magnitude(planes[0], planes[1], img_out);
  img_out += cv::Scalar::all(1); 
  log(img_out, img_out);
  img_out = img_out(cv::Rect(0, 0, img_out.cols & -2, img_out.rows & -2));
  int cx = img_out.cols/2;
  int cy = img_out.rows/2;
  cv::Mat q0(img_out, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  cv::Mat q1(img_out, cv::Rect(cx, 0, cx, cy));  // Top-Right
  cv::Mat q2(img_out, cv::Rect(0, cy, cx, cy));  // Bottom-Left
  cv::Mat q3(img_out, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
  cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);
  cv::normalize(img_out, img_out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  return img_out;
}

cv::Mat CvDft2d::getSpectrumImg(const cv::Mat &complex)
{
  std::vector<cv::Mat> planes(2);
  cv::split(complex, planes);
  cv::Mat img_out(complex.size(), CV_32FC1);
  cv::magnitude(planes[0], planes[1], img_out);
  img_out += cv::Scalar::all(1); 
  log(img_out, img_out);
  img_out = img_out(cv::Rect(0, 0, img_out.cols & -2, img_out.rows & -2));
  int cx = img_out.cols/2;
  int cy = img_out.rows/2;
  cv::Mat q0(img_out, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  cv::Mat q1(img_out, cv::Rect(cx, 0, cx, cy));  // Top-Right
  cv::Mat q2(img_out, cv::Rect(0, cy, cx, cy));  // Bottom-Left
  cv::Mat q3(img_out, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
  cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);
  cv::normalize(img_out, img_out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  return img_out;
}

cv::Mat CvDft2d::getRealMat()
{
  cv::Mat planes[2];
  cv::split(complex_, planes);
  return planes[0];
}

cv::Mat CvDft2d::getImagMat()
{
  cv::Mat planes[2];
  cv::split(complex_, planes);
  return planes[1];
}

cv::Mat CvDft2d::getComplex()
{
  return complex_;
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
  cv::idft(complex_, inv_img_, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
}

cv::Mat CvIDft2d::getInvImg()
{
  cv::normalize(inv_img_, inv_img_, 0, 255, CV_MINMAX, CV_8UC1);
  return inv_img_.clone();
}



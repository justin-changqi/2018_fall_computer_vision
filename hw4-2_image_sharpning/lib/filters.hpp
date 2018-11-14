#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

enum class Method{LAPLACIAN = 0 , SOBEL=1};

class Filter
{
  public:
  Filter(cv::Mat &src_img, Method method);
  cv::Mat getFilteredImg();
  
  private:
  std::vector<cv::Mat> masks;
  cv::Mat src_img;
  cv::Mat pad_img;
  Method method;
  double det_d;
  cv::Mat setMask();
  cv::Mat applyMask(cv::Mat &src_img, cv::Mat &mask);
  void addPadding(cv::Mat &src_img, cv::Mat &mask);
  cv::Mat sumImgs(std::vector<cv::Mat> img_lists);
  int det(cv::Mat &mat);
  cv::Mat refineImg(cv::Mat &src);
  cv::Mat Normalize(cv::Mat &src);
  cv::Mat Constrain(cv::Mat &src);
};
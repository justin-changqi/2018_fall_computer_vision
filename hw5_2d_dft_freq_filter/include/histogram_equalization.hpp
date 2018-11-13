#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

class HistogramEq
{
  public:
  cv::Mat src;
  std::vector<int> histogram;
  std::vector<int> cdf;
  std::vector<int> v_map;
  int cdf_min;
  int L;
  int pixel_num;
  HistogramEq(cv::Mat &src_img, int L);
  std::vector<int> getHistogram(cv::Mat &src_img, int L);
  std::vector<int> getCDF();
  int getHv(int v);
  std::vector<int> getEqHistofram();
  void ComputeVmap(std::vector<int> &v_map);
  cv::Mat getEqImage(cv::Mat &img_src);
  cv::Mat getEqImage();
};
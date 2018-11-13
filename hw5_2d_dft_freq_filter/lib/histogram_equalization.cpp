#include "histogram_equalization.hpp"

HistogramEq::HistogramEq(cv::Mat &src_img, int L=256)
{
  this->histogram = this->getHistogram(src_img, L);
  this->cdf = this->getCDF();
  this->L = L;
  this->pixel_num = src_img.cols * src_img.rows;
  this->src = src_img.clone();
  this->ComputeVmap(this->v_map);
}

std::vector<int> HistogramEq::getHistogram(cv::Mat &src_img, int L=256)
{
  std::vector<int> his(L, 0.0);
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++)
    {
      his[src_img.at<uint8_t>(i, j)] += 1.0;
    }
  }
  return his;
}

std::vector<int> HistogramEq::getCDF()
{
  std::vector<int> his_src = this->histogram;
  std::vector<int> cdf(his_src.size());
  int cdf_count = his_src[0];
  int cdf_min = 0;
  for (int i = 0; i < his_src.size(); i++)
  {
    cdf[i] = cdf_count;
    if (cdf_min == 0 && cdf_count != 0) cdf_min = cdf_count;
    cdf_count += his_src[i];
  }
  this->cdf_min = cdf_min;
  return cdf;
}

int HistogramEq::getHv(int v)
{
  return ((double)(this->cdf[v] - this->cdf_min) / (double)(this->pixel_num)) * (this->L - 1); 
}

std::vector<int> HistogramEq::getEqHistofram()
{
  std::vector<int> eq_his(this->histogram.size());
  for (int i = 0; i < eq_his.size(); i++)
  {
    eq_his[this->getHv(i)] = this->histogram[i];
  }
  return eq_his;
}

void HistogramEq::ComputeVmap(std::vector<int> &v_map)
{
  v_map.resize(this->histogram.size());
  for (int i = 0; i < this->histogram.size(); i++)
  {
    v_map[i] = this->getHv(i);
  }
}

cv::Mat HistogramEq::getEqImage(cv::Mat &img_src)
{
  cv::Mat eq_img(img_src.rows, img_src.cols, CV_8UC1);
  for (int i = 0; i < img_src.rows; i++)
  {
    for (int j = 0; j < img_src.cols; j++)
    {
      eq_img.at<char>(i, j) = this->v_map[img_src.at<uint8_t>(i, j)];
    }
  }
  return eq_img;
}

cv::Mat HistogramEq::getEqImage()
{
  cv::Mat img_src = this->src;
  cv::Mat eq_img(img_src.rows, img_src.cols, CV_8UC1);
  for (int i = 0; i < img_src.rows; i++)
  {
    for (int j = 0; j < img_src.cols; j++)
    {
      eq_img.at<char>(i, j) = this->v_map[img_src.at<uint8_t>(i, j)];
    }
  }
  return eq_img;
}
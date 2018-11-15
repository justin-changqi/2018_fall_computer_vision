#include "freq_filters.hpp"

namespace fdf {
  void applyFilter(const cv::Mat &src, const cv::Mat &mask, cv::Mat &dst)
  { 
    cv::Mat mask_shift = mask.clone();
    int cx = mask_shift.cols/2;
    int cy = mask_shift.rows/2;
    cv::Mat q0(mask_shift, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(mask_shift, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(mask_shift, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(mask_shift, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    for (int i = 0; i < src.rows; i++)
    {
      for (int j = 0; j < src.cols; j++)
      {
        dst.at<cv::Vec2f>(i, j)[0] =  src.at<cv::Vec2f>(i, j)[0] * mask_shift.at<float>(i, j);
        dst.at<cv::Vec2f>(i, j)[1] =  src.at<cv::Vec2f>(i, j)[1] * mask_shift.at<float>(i, j);
      }
    }
  }

  cv::Mat idealLpf(const cv::Mat &src, double d0)
  {
    return getILpfKernel(d0, src.cols, src.rows);
  }

  cv::Mat getILpfKernel(double d0, int width, int height)
  {
    cv::Mat out_img(height, width, CV_32FC1); 
    int cx =  out_img.cols / 2;
    int cy =  out_img.rows / 2;
    for (int i = 0; i < out_img.rows; i++)
    {
      for (int j = 0; j < out_img.cols; j++)
      {
        if (hypot(cx-j, cy-i) <= d0)
        {
          out_img.at<float>(i, j) = 1.;
        }
        else 
        {
          out_img.at<float>(i, j) = 0.;
        }
      }
    }
    return out_img;
  }
  
  cv::Mat gaussianLpf(const cv::Mat &src, double d0)
  {
    return getGLpfKernel(d0, src.cols, src.rows);
  }

  cv::Mat getGLpfKernel(double d0, int width, int height)
  {
    cv::Mat out_img(height, width, CV_32FC1); 
    int cx =  out_img.cols / 2;
    int cy =  out_img.rows / 2;
    for (int i = 0; i < out_img.rows; i++)
    {
      for (int j = 0; j < out_img.cols; j++)
      {
        float d2uv = pow(cx-j, 2)+pow(cy-i, 2);
        out_img.at<float>(i, j) = exp(-d2uv/pow(d0, 2));
      }
    }
    return out_img;
  }
}

#include "geo_tf_chessboard.hpp"

geoTfChessboard::geoTfChessboard(const cv::Mat &chessboard_src, 
                                 const cv::Mat &chessboard_dst,
                                 const cv::Size patternsize)
{
  std::vector<cv::Point2f> corners_src;
  center_pt_ = cv::Point2f(chessboard_src.cols/2, chessboard_src.rows/2);
  bool patternfound_src = findChessboardCorners(chessboard_src, 
                                            patternsize, 
                                            corners_src,
                                            cv::CALIB_CB_ADAPTIVE_THRESH + 
                                            cv::CALIB_CB_NORMALIZE_IMAGE + 
                                            cv::CALIB_CB_FAST_CHECK);
  std::vector<cv::Point2f> corners_dst;
  bool patternfound_dst = findChessboardCorners(chessboard_dst, 
                                            patternsize, 
                                            corners_dst,
                                            cv::CALIB_CB_ADAPTIVE_THRESH + 
                                            cv::CALIB_CB_NORMALIZE_IMAGE + 
                                            cv::CALIB_CB_FAST_CHECK);
  chessboard_draw_ = chessboard_dst.clone();
  drawChessboardCorners(chessboard_draw_, patternsize, 
                        cv::Mat(corners_dst), patternfound_dst);
  this->computeDistortionParam(corners_src, corners_dst);
  // std::cout << corners_dst << std::endl;
  std::cout << k_ << std::endl;
}

cv::Mat geoTfChessboard::getDistoredImg(const cv::Mat &src)
{
  // cv::Mat out = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
  cv::Mat out = cv::Mat::zeros(1000, 1000, CV_8UC1);
  for (int y = 0; y < src.rows; y++)
  {
    for (int x = 0; x < src.cols; x++)
    {
      double r = hypot(center_pt_.y - y, 
                       center_pt_.x - x);
      double dis = 1+pow(r, 2)*k_(0)+pow(r, 4)*k_(1)+pow(r, 6)*k_(2);
      int disx = x*dis;
      int disy = y*dis;
      // std::cout << disy << std::endl;
      out.at<char>(disy, disx) = src.at<char>(y, x);
    }
  }
  return out;
}

void geoTfChessboard::computeDistortionParam(
                    const std::vector<cv::Point2f> orig_pt,
                    const std::vector<cv::Point2f> distr_pt)
{
  int size = orig_pt.size();
  Eigen::MatrixXf A(size*2, 3);
  Eigen::VectorXf b(size*2);
  for (int i = 0; i < size; i++)
  {
    double r = hypot(center_pt_.y - orig_pt[i].y, 
                     center_pt_.x - orig_pt[i].x);
    int mat_index = i*2;
    A(mat_index, 0) = pow(r, 2);
    A(mat_index, 1) = pow(r, 4);
    A(mat_index, 2) = pow(r, 6);
    A(mat_index+1, 0) =  A(mat_index, 0);
    A(mat_index+1, 1) =  A(mat_index, 1);
    A(mat_index+1, 2) =  A(mat_index, 2);
    b(mat_index) = distr_pt[i].x / orig_pt[i].x - 1;
    b(mat_index+1) = distr_pt[i].y / orig_pt[i].y - 1;
  }
  k_ = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}
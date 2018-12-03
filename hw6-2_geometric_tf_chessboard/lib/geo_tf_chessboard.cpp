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
  chessboard_src_draw_ = chessboard_src.clone();
  chessboard_dst_draw_ = chessboard_dst.clone();
  drawChessboardCorners(chessboard_src_draw_, patternsize, 
                        cv::Mat(corners_src), patternfound_src);
  drawChessboardCorners(chessboard_dst_draw_, patternsize, 
                        cv::Mat(corners_dst), patternfound_dst);
  this->computeDistortionParam(corners_src, corners_dst);;
  std::cout << k_ << std::endl;
  std::cout << p_ << std::endl;
}

// cv::Mat geoTfChessboard::getDistoredImg(const cv::Mat &src)
// {
//   cv::Mat out = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
//   // cv::Mat out = cv::Mat::zeros(2000, 2000, CV_8UC1);
//   for (int y = 0; y < src.rows; y++)
//   {
//     for (int x = 0; x < src.cols; x++)
//     {
//       double s_orig_x = x - center_pt_.x;
//       double s_orig_y = y - center_pt_.y;
//       double r_2 = pow(s_orig_x, 2) + pow(s_orig_y, 2);
//       double r_4 = pow(r_2, 2);
//       double dis = 1-r_2*k_(0)+r_4*k_(1);
//       int disx = s_orig_x*dis+center_pt_.x;
//       int disy = s_orig_y*dis+center_pt_.y;
//       // std::cout << disy << std::endl;
//       out.at<char>(disy, disx) = src.at<char>(y, x);
//     }
//   }
//   return out;
// }

// void geoTfChessboard::computeDistortionParam(
//                     const std::vector<cv::Point2f> orig_pt,
//                     const std::vector<cv::Point2f> distr_pt)
// {
//   int size = orig_pt.size();
//   Eigen::MatrixXf A(size*2, 2);
//   Eigen::VectorXf b(size*2);
//   for (int i = 0; i < size; i++)
//   {
//     double s_orig_x = orig_pt[i].x - center_pt_.x;
//     double s_orig_y = orig_pt[i].y - center_pt_.y;
//     double s_distro_x = distr_pt[i].x - center_pt_.x;
//     double s_distro_y = distr_pt[i].y - center_pt_.y;
//     double r_2 = pow(s_orig_x, 2) + pow(s_orig_y, 2);
//     int mat_index = i*2;
//     A(mat_index, 0) = r_2;
//     A(mat_index, 1) = pow(r_2, 2);
//     // A(mat_index, 2) = pow(r, 6);
//     A(mat_index+1, 0) =  A(mat_index, 0);
//     A(mat_index+1, 1) =  A(mat_index, 1);
//     // A(mat_index+1, 2) =  A(mat_index, 2);
//     b(mat_index) = s_distro_x / s_orig_x - 1;
//     b(mat_index+1) = s_distro_y / s_orig_y - 1;
//   }
//   // SVD
//   // std::cout << A << std::endl;
//   // k_ = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
//   // Normal
//   k_ = (A.transpose() * A).ldlt().solve(A.transpose() * b);
//   // QR
//   // k_ = A.colPivHouseholderQr().solve(b);
//   // k_ = Eigen::VectorXf::Zero(3);
//   k_(0) = 0.000015;
//   k_(1) = 0.00000000015;
//   // k_(1) = 0;
//   // // k_(2) = 0.0000000000001;
//   // k_(2) = 0;
// }

cv::Mat geoTfChessboard::getDistoredImg(const cv::Mat &src)
{
  cv::Mat out = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
  // cv::Mat out = cv::Mat::zeros(2000, 2000, CV_8UC1);
  for (int y = 0; y < src.rows; y++)
  {
    for (int x = 0; x < src.cols; x++)
    {
      double s_orig_x = x - center_pt_.x;
      double s_orig_y = y - center_pt_.y;
      double r_2 = pow(s_orig_x, 2) + pow(s_orig_y, 2);
      double r_4 = pow(r_2, 2);
      double dis = 1-r_2*k_(0)+r_4*k_(1);
      int disx = s_orig_x*dis+center_pt_.x+(p_(0)*(r_2+2*s_orig_x))+
                  2*p_(1)*s_orig_x*s_orig_y;
      int disy = s_orig_y*dis+center_pt_.y+(2*p_(0)*s_orig_x*s_orig_y)+
                  (p_(1)*(r_2+2*s_orig_y));
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
  Eigen::MatrixXf A(size*2, 2);
  Eigen::VectorXf b(size*2);
  Eigen::MatrixXf p_A(size*2, 2);
  Eigen::VectorXf p_b(size*2);
  for (int i = 0; i < size; i++)
  {
    double s_orig_x = orig_pt[i].x - center_pt_.x;
    double s_orig_y = orig_pt[i].y - center_pt_.y;
    double s_distro_x = distr_pt[i].x - center_pt_.x;
    double s_distro_y = distr_pt[i].y - center_pt_.y;
    double r_2 = pow(s_orig_x, 2) + pow(s_orig_y, 2);
    int mat_index = i*2;
    A(mat_index, 0) = r_2;
    A(mat_index, 1) = pow(r_2, 2);
    // A(mat_index, 2) = pow(r, 6);
    A(mat_index+1, 0) =  A(mat_index, 0);
    A(mat_index+1, 1) =  A(mat_index, 1);
    // A(mat_index+1, 2) =  A(mat_index, 2);
    b(mat_index) = s_distro_x / s_orig_x - 1;
    b(mat_index+1) = s_distro_y / s_orig_y - 1;
    // calculate P
    p_A(mat_index, 0) = 2*s_orig_x*s_orig_y;
    p_A(mat_index, 1) = r_2+2*pow(s_orig_x, 2);
    p_A(mat_index+1, 0) = r_2+2*pow(s_orig_y, 2);
    p_A(mat_index+1, 1) = 2*s_orig_x*s_orig_y;
    p_b(mat_index) = s_distro_x - s_orig_x;
    p_b(mat_index+1) = s_distro_y - s_orig_y; 
  }
  // SVD
  // std::cout << A << std::endl;
  // k_ = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  // Normal
  // k_ = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  // p_ = (p_A.transpose() * p_A).ldlt().solve(p_A.transpose() * p_b);
  // QR
  k_ = A.colPivHouseholderQr().solve(b);
  p_ = p_A.colPivHouseholderQr().solve(p_b);
}
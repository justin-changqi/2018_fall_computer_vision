#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>

// Ref. https://math.stackexchange.com/questions/302093/how-to-calculate-the-lens-distortion-coefficients-with-a-known-displacement-vect
// Ref. https://eigen.tuxfamily.org/dox/group__LeastSquares.html

class geoTfChessboard
{
  public:
  cv::Mat chessboard_src_draw_;
  cv::Mat chessboard_dst_draw_;
  geoTfChessboard(const cv::Mat &chessboard_src, 
                  const cv::Mat &chessboard_dst,
                  const cv::Size patternsize);
  cv::Mat getDistoredImg(const cv::Mat &src);

  private:
  cv::Point2f center_pt_;
  Eigen::MatrixXf k_;
  Eigen::MatrixXf p_;
  void computeDistortionParam(const std::vector<cv::Point2f> orig_pt,
                              const std::vector<cv::Point2f> distr_pt);
};
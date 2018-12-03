#include "image_io.hpp"
#include "geo_tf_chessboard.hpp"

int main(int argc, char **argv)
{
  cv::Mat chessboard(256, 256, CV_8UC1);
  loadRawFile(chessboard, "../images/chessboard_256.raw", 256, 256);
  cv::Mat chessboard_tf(256, 256, CV_8UC1);
  loadRawFile(chessboard_tf, "../images/chessboard_distorted_256.raw", 256, 256);
  geoTfChessboard gtc = geoTfChessboard(chessboard, chessboard_tf, 
                                        cv::Size(7, 7));
  cv::Mat dis_img = gtc.getDistoredImg(chessboard);
  showImage("chessboard", dis_img);
  showImage("chessboard_dis", chessboard_tf);
  // saveImage(chessboard, "../result_img/", "chessboard");
  cv::waitKey(0);
  return 0;
}
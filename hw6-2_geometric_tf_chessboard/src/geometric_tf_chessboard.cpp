#include "image_io.hpp"
#include "geo_tf_chessboard.hpp"

int main(int argc, char **argv)
{
  cv::Mat chessboard(256, 256, CV_8UC1);
  loadRawFile(chessboard, "../images/chessboard_256.raw", 256, 256);
  cv::Mat chessboard_tf(256, 256, CV_8UC1);
  loadRawFile(chessboard_tf, "../images/chessboard_distorted_256.raw", 256, 256);
  cv::Mat owl_img(256, 256, CV_8UC1);
  loadRawFile(owl_img, "../images/owl.raw", 256, 256);
  geoTfChessboard gtc = geoTfChessboard(chessboard, chessboard_tf, 
                                        cv::Size(7, 7));
  cv::Mat distored_tf = gtc.getDistoredImg(owl_img);
  showImage("chessboard_draw_src", gtc.chessboard_src_draw_);
  showImage("chessboard_draw_dst", gtc.chessboard_dst_draw_);
  showImage("chessboard_dis", distored_tf);
  saveImage(gtc.chessboard_src_draw_, "../result_img/", 
            "chessboard_draw_src");
  saveImage(gtc.chessboard_dst_draw_, "../result_img/", 
            "chessboard_draw_dst");
  saveImage(distored_tf, "../result_img/", "distored_tf");
  cv::waitKey(0);
  return 0;
}
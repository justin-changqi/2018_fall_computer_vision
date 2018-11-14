#include "image_io.hpp"

int main(int argc, char **argv)
{
  cv::Mat src(256, 256, CV_8UC1);
  loadRawFile(src, "../images/cat_bright.raw", 256, 256);
  showImage("src", src);
  saveImage(src, "../result_img/", "src");
  cv::waitKey(0);
  return 0;
}
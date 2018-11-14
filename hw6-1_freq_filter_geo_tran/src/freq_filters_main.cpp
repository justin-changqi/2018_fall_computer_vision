#include "image_io.hpp"
#include "cv_dft_2d.hpp"

int main(int argc, char **argv)
{
  cv::Mat fox_src = loadRawFile("../images/fox.raw", 380, 284);
  CvDft2d fox_dft(fox_src);
  CvIDft2d fox_idft(fox_dft.getComplex());
  cv::Mat fix_idft_img =  fox_idft.getInvImg();
  showImage("fox_src", fox_src);
  showImage("fox_idft",fix_idft_img);
  saveImage(fox_src, "../result_img/", "fox_src");
  cv::waitKey(0);
  return 0;
}
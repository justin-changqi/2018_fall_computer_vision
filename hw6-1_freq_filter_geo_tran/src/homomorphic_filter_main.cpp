#include "image_io.hpp"
#include "cv_dft_2d.hpp"
#include "freq_filters.hpp"

int main(int argc, char **argv)
{
  cv::Mat wolf_src = loadRawFile("../images/wolf_dark.raw", 256, 256);
  CvDft2d wolf_dft(wolf_src);
  cv::Mat wolf_dft_i = wolf_dft.getComplex();
  cv::Mat wolf_sp = wolf_dft.getSpectrumImg();
  saveImage(wolf_src, "../result_img/", "wolf_src");
  saveImage(wolf_sp, "../result_img/", "wolf_sp");
  showImage("fox_src",wolf_src);
  showImage("fox_sp",wolf_sp);
  // Homomorphic filter
  cv::Mat homo_mask = fdf::HomomorphicLpf(wolf_dft_i, 1, 10, 2, 100);
  cv::Mat homo_filtered_complex(wolf_dft_i.size(), CV_32FC2);
  fdf::applyFilter(wolf_dft_i, homo_mask, homo_filtered_complex);
  cv::Mat homo_filter_spectrum  = CvDft2d::getSpectrumImg(homo_filtered_complex); 
  CvIDft2d homo_idft(homo_filtered_complex);
  cv::Mat homo_idft_img = homo_idft.getInvImg();
  cv::normalize(homo_mask, homo_mask, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  showImage("mask", homo_mask);
  showImage("filtered sp", homo_filter_spectrum);
  showImage("idft", homo_idft_img);
  saveImage(homo_mask, "../result_img/1.c/", "wolf_homomorphic_mask");
  saveImage(homo_filter_spectrum, "../result_img/1.c/", "wolf_homomorphic_sp_filtered");
  saveImage(homo_idft_img, "../result_img/1.c/", "wolf_homomorphic_idft");
  cv::waitKey(0);
  return 0;
}
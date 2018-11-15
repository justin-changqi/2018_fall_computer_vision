#include "image_io.hpp"
#include "cv_dft_2d.hpp"
#include "freq_filters.hpp"

int main(int argc, char **argv)
{
  cv::Mat fox_src = loadRawFile("../images/fox.raw", 380, 284);
  CvDft2d fox_dft(fox_src);
  cv::Mat fox_dft_i = fox_dft.getComplex();
  cv::Mat fox_sp = fox_dft.getSpectrumImg();
  cv::Mat fox_ilpf_15(fox_dft_i.rows, fox_dft_i.cols, CV_32FC2);
  cv::Mat ilpf_15 =  fdf::idealLpf(fox_dft.getComplex(), 15);
  fdf::applyFilter(fox_dft_i, ilpf_15, fox_ilpf_15);
  cv::Mat fox_ilpf_15_sp = CvDft2d::getSpectrumImg(fox_ilpf_15); 
  CvIDft2d fox_idft_15(fox_ilpf_15);
  cv::Mat fox_idft_img_15 =  fox_idft_15.getInvImg();
  cv::Mat fox_ilpf_50(fox_dft_i.rows, fox_dft_i.cols, CV_32FC2);
  cv::Mat ilpf_50 =  fdf::idealLpf(fox_dft.getComplex(), 50);
  fdf::applyFilter(fox_dft_i, ilpf_50, fox_ilpf_50);
  cv::Mat fox_ilpf_50_sp = CvDft2d::getSpectrumImg(fox_ilpf_50); 
  CvIDft2d fox_idft_50(fox_ilpf_50);
  cv::Mat fox_idft_img_50 =  fox_idft_50.getInvImg();
  showImage("fox_src", fox_src);
  showImage("fox_dft_sp",fox_sp);
  showImage("fox_idft_15",fox_idft_img_15);
  showImage("fox_dft_filter_sp_15",fox_ilpf_15_sp);
  showImage("fox_idft_50",fox_idft_img_50);
  showImage("fox_dft_filter_sp_50",fox_ilpf_50_sp);
  cv::normalize(ilpf_15, ilpf_15, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::normalize(ilpf_50, ilpf_50, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  saveImage(fox_src, "../result_img/", "fox_src");
  saveImage(fox_sp, "../result_img/", "fox_sp");
  saveImage(ilpf_15, "../result_img/1.a/", "ilpf_15");
  saveImage(fox_idft_img_15, "../result_img/1.a/", "fox_filterd_ilpf_15");
  saveImage(ilpf_50, "../result_img/1.a/", "ilpf_50");
  saveImage(fox_idft_img_50, "../result_img/1.a/", "fox_filterd_ilpf_50");
  cv::waitKey(0);
  return 0;
}
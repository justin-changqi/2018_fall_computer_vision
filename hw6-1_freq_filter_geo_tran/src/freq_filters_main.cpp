#include "image_io.hpp"
#include "cv_dft_2d.hpp"
#include "freq_filters.hpp"


int main(int argc, char **argv)
{
  cv::Mat fox_src = loadRawFile("../images/fox.raw", 380, 284);
  CvDft2d fox_dft(fox_src);
  cv::Mat fox_dft_i = fox_dft.getComplex();
  cv::Mat fox_sp = fox_dft.getSpectrumImg();
  saveImage(fox_src, "../result_img/", "fox_src");
  saveImage(fox_sp, "../result_img/", "fox_sp");
  showImage("fox_src",fox_src);
  showImage("fox_sp",fox_sp);

  // ideal filter
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
  cv::normalize(ilpf_15, ilpf_15, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::normalize(ilpf_50, ilpf_50, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  saveImage(ilpf_15, "../result_img/1.a/", "ideal_lpf_15_sp");
  saveImage(fox_idft_img_15, "../result_img/1.a/", "ideal_lpf_15_idft");
  saveImage(ilpf_50, "../result_img/1.a/", "ideal_lpf_50_sp");
  saveImage(fox_idft_img_50, "../result_img/1.a/", "ideal_lpf_50_idft");

  // Gaussian LPF 
  cv::Mat fox_g_lpf_15(fox_dft_i.rows, fox_dft_i.cols, CV_32FC2);
  cv::Mat glpf_15 =  fdf::gaussianLpf(fox_dft.getComplex(), 15);
  fdf::applyFilter(fox_dft_i, glpf_15, fox_g_lpf_15);
  cv::Mat fox_g_lpf_15_sp = CvDft2d::getSpectrumImg(fox_g_lpf_15); 
  CvIDft2d fox_g_idft_15(fox_g_lpf_15);
  cv::Mat fox_g_idft_img_15 =  fox_g_idft_15.getInvImg();
  cv::Mat fox_g_lpf_50(fox_dft_i.rows, fox_dft_i.cols, CV_32FC2);
  cv::Mat glpf_50 =  fdf::gaussianLpf(fox_dft.getComplex(), 50);
  fdf::applyFilter(fox_dft_i, glpf_50, fox_g_lpf_50);
  cv::Mat fox_g_ilpf_50_sp = CvDft2d::getSpectrumImg(fox_g_lpf_50); 
  CvIDft2d fox_g_idft_50(fox_g_lpf_50);
  cv::Mat fox_g_idft_img_50 =  fox_g_idft_50.getInvImg();
  cv::normalize(glpf_15, glpf_15, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::normalize(glpf_50, glpf_50, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  saveImage(glpf_15, "../result_img/1.a/", "gaussion_lpf_15_sp");
  saveImage(fox_g_idft_img_15, "../result_img/1.a/", "gaussion_lpf_15_idft");
  saveImage(glpf_50, "../result_img/1.a/", "gaussion_lpf_50_sp");
  saveImage(fox_g_idft_img_50, "../result_img/1.a/", "gaussion_lpf_50_idft");

  showImage("fox_g_idft_15",fox_g_idft_img_15);
  showImage("fox_g_dft_filter_sp_15",fox_g_lpf_15_sp);
  showImage("fox_g_idft_50",fox_g_idft_img_50);
  showImage("fox_g_dft_filter_sp_50",fox_g_ilpf_50_sp);

  cv::waitKey(0);
  return 0;
}
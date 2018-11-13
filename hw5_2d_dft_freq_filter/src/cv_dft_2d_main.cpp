#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include "cv_dft_2d.hpp"

void loadRawFile(cv::Mat &dst_img, const std::string file_path, int width, int height);
void showImage(std::string win_name, cv::Mat &show_img);
void saveImage(cv::Mat &img, const std::string folder, std::string file_name);

void loadRawFile(cv::Mat &dst_img, const std::string file_path, int width, int height)
{
  std::FILE* f = std::fopen(file_path.c_str(), "rb");
  // std::vector<char> buf(width*height);    // char is trivally copyable
  unsigned char buf[width][height];
  std::fread(&buf[0], sizeof buf[0], width*height, f);
  for (int i = 0; i < dst_img.rows; i++)
  {
    for (int j = 0; j < dst_img.cols; j++)
    {
      dst_img.at<char>(i, j) = buf[i][j];
    }
  }
  std::fclose(f);
}

void showImage(std::string win_name, cv::Mat &show_img)
{
  static int win_move_x = 50;
  static int win_move_y = 50;
  cv::namedWindow(win_name, 0);
  cv::resizeWindow(win_name, show_img.cols, show_img.rows);
  cv::moveWindow(win_name, win_move_x, win_move_y);
  cv::imshow(win_name, show_img); //display Image
  win_move_x +=  show_img.cols;
  if (win_move_x > 1920-256)
  {
    win_move_x = 50;
    win_move_y += (show_img.rows+35);
  }
}

void saveImage(cv::Mat &img, const std::string folder, std::string file_name)
{
  std::string save_file = folder + file_name + ".png";
  cv::imwrite(save_file, img);
}

void timeMeasure(bool starting)
{
  using namespace std::chrono; 
  static time_point<_V2::system_clock, nanoseconds> start, stop;
  if (starting)
  {
    start = std::chrono::high_resolution_clock::now(); 
  }
  else
  {
    stop = std::chrono::high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Execute time: " <<  duration.count() << " microseconds" << std::endl;
  }
}

int main(int argc, char **argv)
{
  
  cv::Mat src(512, 512, CV_8UC1);
  loadRawFile(src, "../images/lena512.raw", 512, 512);
  cv::Mat lena_src_n(512, 512, CV_8UC1);
  loadRawFile(lena_src_n, "../images/lena512_noise.raw", 512, 512);
  CvDft2d dft2d(src);
  std::cout << "Start Lena Opencv DFT" << std::endl;
  timeMeasure(true);
  dft2d.computeDft();
  timeMeasure(false);
  cv::Mat dft_result = dft2d.getSpectrumImg();
  CvIDft2d idft2d(dft2d.getRealMat(), dft2d.getImagMat());
  std::cout << "Start Lena Opencv IDFT" << std::endl;
  timeMeasure(true);
  idft2d.computeIDft();
  timeMeasure(false);
  cv::Mat idft_result = idft2d.getInvImg();
  
  CvDft2d dft2d_n(lena_src_n);
  std::cout << "Start Lena Noised Opencv DFT" << std::endl;
  timeMeasure(true);
  dft2d_n.computeDft();
  timeMeasure(false);
  cv::Mat dft_result_n = dft2d_n.getSpectrumImg();
  CvIDft2d idft2d_n(dft2d_n.getRealMat(), dft2d_n.getImagMat());
  std::cout << "Start Lena Noised Opencv IDFT" << std::endl;
  timeMeasure(true);
  idft2d_n.computeIDft();
  timeMeasure(false);
  cv::Mat idft_result_n = idft2d_n.getInvImg();
  showImage("DFT result", dft_result);
  showImage("IDFT result", idft_result);
  showImage("DFT Noised result", dft_result_n);
  showImage("IDFT Noised result", idft_result_n);
  saveImage(dft_result, "../result_img/", "cv_lena_dft");
  saveImage(idft_result, "../result_img/", "cv_lena_idft");
  saveImage(dft_result_n, "../result_img/", "cv_lena_noise_dft");
  saveImage(idft_result_n, "../result_img/", "cv_lena_noise_idft");
  cv::waitKey(0);
  return 0;
}
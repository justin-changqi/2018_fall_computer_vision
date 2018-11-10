#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dft_2d.hpp"

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

int main(int argc, char **argv)
{
  cv::Mat src(512, 512, CV_8UC1);
  loadRawFile(src, "../images/lena512.raw", 512, 512);
  Dft2d dft2d(src);
  cv::Mat dft_result = dft2d.getDftImg();
  showImage("DFT result", dft_result);
  // saveImage(src, "../result_img/", "p1_src");
  cv::waitKey(0);
  return 0;
}
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
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

cv::Mat getFilterMask(const cv::Mat &img, int r)
{
  int cx = img.cols / 2;
  int cy = img.rows / 2;
  cv::Mat out_img(img.rows, img.cols, CV_64FC1);
  for (int i = 0; i < img.rows; i++)
  {
    for (int j = 0; j < img.rows; j++)
    {
      int dis = hypot(i-cx, j-cy);
      if (dis > r)
      {
        if(img.at<uint8_t>(i, j) > 150)
        {
          out_img.at<double>(i, j) = 0;
        }
        else
        {
          out_img.at<double>(i, j) = 1;
        }
      }
      else
      {
        out_img.at<double>(i, j) = 1;
      }
    }
  }
  return out_img;
}

cv::Mat filterNormalize(cv::Mat &img_in)
{
  cv::Mat out_img(img_in.rows, img_in.cols, CV_8UC1);
  cv::normalize(img_in, out_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::bitwise_not(out_img, out_img);
  return out_img;
}

int main(int argc, char **argv)
{
  cv::Mat fingerpoint_src(256, 256, CV_8UC1);
  loadRawFile(fingerpoint_src, "../images/fingerpoint.raw", 256, 256);
  CvDft2d dft2d_src(fingerpoint_src);
  dft2d_src.computeDft();
  cv::Mat dft2d_fingerpoint_sp = dft2d_src.getSpectrumImg();
  cv::Mat sp_filter = getFilterMask(dft2d_fingerpoint_sp, 30);
  cv::Mat sp_filter_nor = filterNormalize(sp_filter);
  showImage("finger pointsrc result", fingerpoint_src);
  showImage("DFT src result", dft2d_fingerpoint_sp);
  showImage("filter", sp_filter_nor);

  saveImage(fingerpoint_src, "../result_img/", "fingerpoint_src");
  saveImage(dft2d_fingerpoint_sp, "../result_img/", "fingerpoint_source_sp");
  saveImage(sp_filter_nor, "../result_img/", "filter");
  cv::waitKey(0);
  return 0;
}
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

cv::Mat addPadding(cv::Mat &src_img, cv::Mat &mask)
{
  int pad_x = (mask.cols-1)/2;
  int pad_y = (mask.rows-1)/2;
  cv::Mat out_img(src_img.rows+2*pad_y, 
                  src_img.cols+2*pad_x, CV_8UC1);
  // put src to new image
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++)
    {
      out_img.at<char>(i+pad_y, j+pad_x) = src_img.at<char>(i, j);
    }
  }
  // mirror rows
  for (int i = 0; i <= pad_y; i++)
  {
    out_img.row(pad_y+i).copyTo(out_img.row(pad_y-i));
    out_img.row(out_img.rows-1-pad_y-i).copyTo(out_img.row(out_img.rows-1-pad_y+i));
  }
  // mirror cols
  for (int i = 0; i <= pad_x; i++)
  {
    out_img.col(pad_x+i).copyTo(out_img.col(pad_x-i));
    out_img.col(out_img.cols-1-pad_x-i).copyTo(out_img.col(out_img.cols-1-pad_x+i));
  }
  return out_img;
}

cv::Mat averagingFilter(cv::Mat &src, cv::Mat &mask)
{
  cv::Mat out_img(src.rows, src.cols, CV_8UC1);
  cv::Mat padded_img = addPadding(src, mask);
  int pad_x = (mask.cols-1) / 2;
  int pad_y = (mask.rows-1) / 2;
  // Normalize size
  double nol_size = 0;
  for (int i = 0; i < mask.rows; i++)
  {
    for (int j = 0; j < mask.cols; j++)
    {
      nol_size += mask.at<double>(i, j);
    }
  }
  // apply LPF
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      double px = 0;
      int i_offset = i + pad_y;
      int j_offset = j + pad_x;
      for (int k = i_offset-pad_y; k <= i_offset+pad_y; k++)
      {
        for (int l = j_offset-pad_x; l <= j_offset+pad_x; l++)
        {
          px += padded_img.at<uint8_t>(k, l);
        }
      }
      px /= nol_size;
      out_img.at<uint8_t>(i, j) = px;
    }
  }
  return out_img;
}

void setBoxMask(cv::Mat &mask)
{
  for (int i = 0; i < mask.rows; i++)
  {
    for (int j = 0; j < mask.cols; j++)
    {
      mask.at<double>(i, j) = 1;
    }
  }
}

cv::Mat medianFilter(cv::Mat &src, cv::Mat &mask)
{
  cv::Mat out_img(src.rows, src.cols, CV_8UC1);
  cv::Mat padded_img = addPadding(src, mask);
  int pad_x = (mask.cols-1) / 2;
  int pad_y = (mask.rows-1) / 2;
  int median_inx = (mask.cols * mask.rows) / 2;
  // apply Median Filter
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      double px = 0;
      int i_offset = i + pad_y;
      int j_offset = j + pad_x;
      std::vector<uint8_t> mask_px;
      for (int k = i_offset-pad_y; k <= i_offset+pad_y; k++)
      {
        for (int l = j_offset-pad_x; l <= j_offset+pad_x; l++)
        {
          mask_px.push_back(padded_img.at<uint8_t>(k, l));
        }
      }
      std::sort (mask_px.begin(), mask_px.end());  
      out_img.at<uint8_t>(i, j) = mask_px[median_inx];
    }
  }
  return out_img;
}

int main(int argc, char **argv)
{
  cv::Mat lena_src_n(512, 512, CV_8UC1);
  loadRawFile(lena_src_n, "../images/lena512_noise.raw", 512, 512);
  cv::Mat mask = (cv::Mat_<double>(5,5));
  setBoxMask(mask);
  cv::Mat box_img = averagingFilter(lena_src_n, mask);
  cv::Mat median_img = medianFilter(lena_src_n, mask);
  CvDft2d dft2d_box(box_img);
  dft2d_box.computeDft();
  cv::Mat dft2d_box_result = dft2d_box.getSpectrumImg();
  CvDft2d dft2d_median(median_img);
  dft2d_median.computeDft();
  cv::Mat dft2d_median_result = dft2d_median.getSpectrumImg();
  showImage("DFT Box result", dft2d_box_result);
  showImage("DFT Meadian result", dft2d_median_result);
  saveImage(dft2d_box_result, "../result_img/", "lena_noise_box_dft");
  saveImage(dft2d_median_result, "../result_img/", "lena_noise_median_dft");
  cv::waitKey(0);
  return 0;
}
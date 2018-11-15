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

void applyFilter(const cv::Mat &src, const cv::Mat &mask, cv::Mat &dst)
{ 
  cv::Mat mask_shift = mask.clone();
  int cx = mask_shift.cols/2;
  int cy = mask_shift.rows/2;
  cv::Mat q0(mask_shift, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  cv::Mat q1(mask_shift, cv::Rect(cx, 0, cx, cy));  // Top-Right
  cv::Mat q2(mask_shift, cv::Rect(0, cy, cx, cy));  // Bottom-Left
  cv::Mat q3(mask_shift, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
  cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      dst.at<cv::Vec2f>(i, j)[0] = src.at<cv::Vec2f>(i, j)[0] * mask_shift.at<double>(i, j);
      dst.at<cv::Vec2f>(i, j)[1] = src.at<cv::Vec2f>(i, j)[1] * mask_shift.at<double>(i, j);
    }
  }
}

int main(int argc, char **argv)
{
  cv::Mat fingerpoint_src(256, 256, CV_8UC1);
  loadRawFile(fingerpoint_src, "../images/fingerpoint.raw", 256, 256);
  CvDft2d dft2d_src(fingerpoint_src);
  cv::Mat dft2d_fingerpoint_sp = dft2d_src.getSpectrumImg();
  cv::Mat sp_filter = getFilterMask(dft2d_fingerpoint_sp, 30);
  cv::Mat sp_filter_nor = filterNormalize(sp_filter);
  cv::Mat filtered_complex(sp_filter.size(), CV_32FC2);
  cv::Mat src_dft_i = dft2d_src.getComplex();
  applyFilter(src_dft_i, sp_filter, filtered_complex);
  cv::Mat filtered_sp = CvDft2d::getSpectrumImg(filtered_complex);
  CvIDft2d idft(filtered_complex);
  cv::Mat invert_img =  idft.getInvImg();
  showImage("finger pointsrc result", fingerpoint_src);
  showImage("DFT src result", dft2d_fingerpoint_sp);
  showImage("filter", sp_filter_nor);
  showImage("filtered_sp", filtered_sp);
  showImage("invert_img", invert_img);

  saveImage(fingerpoint_src, "../result_img/", "fingerpoint_src");
  saveImage(dft2d_fingerpoint_sp, "../result_img/", "fingerpoint_source_sp");
  saveImage(sp_filter_nor, "../result_img/", "filter");
  saveImage(filtered_sp, "../result_img/", "filtered_sp");
  saveImage(invert_img, "../result_img/", "invert_img");
  cv::waitKey(0);
  return 0;
}
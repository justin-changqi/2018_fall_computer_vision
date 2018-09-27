#include "grey_level_resolution.hpp"

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height)
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

cv::Mat getQuantizeImage(cv::Mat &src, int num_bit)
{
  cv::Mat img_out(src.rows, src.cols, CV_8UC1);
  char mask = 0xff << (8-num_bit);
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      img_out.at<char>(i, j) = src.at<char>(i, j) & mask;
    }
  }
  return img_out;
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

void showAllImages(std::vector<cv::Mat> &list, std::string prefix)
{
  for (int i = 0; i < list.size(); i++)
  {
     showImage(prefix + " " + std::to_string(i + 1) + " bits", list[i]);
  }
}

double getMSE(cv::Mat &src, cv::Mat &target)
{
  double mse = 0;
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      mse += pow(src.at<char>(i, j) - target.at<char>(i, j), 2);
    }
  }
  return mse/(src.rows * src.cols);
}

int main(int argc, char **argv)
{
  cv::Mat lena_src(256, 256, CV_8UC1);
  cv::Mat baboon_src(256, 256, CV_8UC1);
  loadRawFile(lena_src, "../images/lena_256.raw", 256, 256);
  loadRawFile(baboon_src, "../images/baboon_256.raw", 256, 256);
  std::vector<cv::Mat> lena_result_list;
  std::vector<cv::Mat> baboon_result_list;
  // get quantize data from 1 bit to 8 bits
  std::cout << "Lena MSE:" << std::endl;
  for (int i = 1; i <= 8; i++)
  {
    lena_result_list.push_back(getQuantizeImage(lena_src, i));
    baboon_result_list.push_back(getQuantizeImage(baboon_src, i));
  }
  // calculate MSE and PSNR
  for (int i = 0; i < 8; i++)
  {
    std::cout << "  lena " << i+1 << " bits: " << getMSE(lena_src, lena_result_list[i]) << std::endl;
  }
  std::cout << "Baboon MSE:" << std::endl;
  for (int i = 0; i < 8; i++)
  {
    std::cout << "  baboon " << i+1 << " bits: " << getMSE(baboon_src, baboon_result_list[i]) << std::endl;
  }
  showAllImages(lena_result_list, "lena");
  showAllImages(baboon_result_list, "baboon");
  cv::waitKey(0);
  return 0;
}
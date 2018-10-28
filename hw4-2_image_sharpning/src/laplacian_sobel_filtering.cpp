#include "laplacian_sobel_filtering.hpp"

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

void saveImage(cv::Mat &img, std::string folder, std::string file_name)
{
  std::string save_file = folder + file_name + ".png";
  cv::imwrite(save_file, img);
}

int main(int argc, char **argv)
{
  cv::Mat src(512, 512, CV_8UC1);
  cv::Mat src_n(512, 512, CV_8UC1);
  loadRawFile(src, "../images/lena512.raw", 512, 512);
  loadRawFile(src_n, "../images/lena512_noise.raw", 512, 512);
  Filter laplacian(src, Method::LAPLACIAN);
  Filter laplacian_n(src_n, Method::LAPLACIAN);
  cv::Mat LPC = laplacian.getFilteredImg();
  cv::Mat LPC_n = laplacian_n.getFilteredImg();
  // Filter sobel(src, Method::SOBEL);
  showImage("lena src", src);
  showImage("lena laplacian", LPC);
  showImage("lena noise src", src_n);
  showImage("lena noise laplacian", LPC_n);
  saveImage(src, "../result_img/", "lena_src");
  saveImage(src_n, "../result_img/", "lena_noise_src");
  cv::waitKey(0);
  return 0;
}
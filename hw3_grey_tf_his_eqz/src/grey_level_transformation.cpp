#include "grey_level_transformation.hpp"

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

void PowerLawTransformation(cv::Mat &src, cv::Mat &dst, double gamma)
{
  double c = 1.0;
  double L = 255;
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      double src_value = src.at<unsigned char>(i, j);
      double dst_value = L * c * pow(src_value/L, gamma);
      // dst_value = std::max(0.0, dst_value);
      // dst_value = std::min(1.0, dst_value);
      dst.at<char>(i, j) = (char) dst_value;
    }
  }
}

void showAllImages(std::vector<cv::Mat> &list, std::string prefix)
{
  for (int i = 0; i < list.size(); i++)
  {
    std::string gamma = std::to_string(GAMMAS[i]);
    gamma.erase ( gamma.find_last_not_of('0') + 2, std::string::npos );
    showImage(prefix + " " + gamma + "gamma", list[i]);
  }
}

void saveAllImages(std::vector<cv::Mat> &list, std::string floder, std::string prefix)
{
  for (int i = 0; i < list.size(); i++)
  {
    std::string gamma = std::to_string(GAMMAS[i]);
    gamma.erase ( gamma.find_last_not_of('0') + 2, std::string::npos );
    std::string save_file = floder + prefix + gamma + ".png";
    cv::imwrite(save_file, list[i]);
  }
}

int main(int argc, char **argv)
{
  cv::Mat cat_b_src(256, 256, CV_8UC1);
  cv::Mat cat_d_src(256, 256, CV_8UC1);
  loadRawFile(cat_b_src, "../images/cat_bright.raw", 256, 256);
  loadRawFile(cat_d_src, "../images/cat_dark.raw", 256, 256);
  std::vector<cv::Mat> cat_b_img_lst;
  std::vector<cv::Mat> cat_d_img_lst;
  for (int i = 0; i < sizeof(GAMMAS)/sizeof(double); i++)
  {
    cv::Mat cat_b_transtormed(256, 256, CV_8UC1);
    cv::Mat cat_d_transtormed(256, 256, CV_8UC1);
    PowerLawTransformation(cat_b_src, cat_b_transtormed, GAMMAS[i]);
    PowerLawTransformation(cat_d_src, cat_d_transtormed, GAMMAS[i]);
    cat_b_img_lst.push_back(cat_b_transtormed);
    cat_d_img_lst.push_back(cat_d_transtormed);
  }
  // showAllImages(cat_b_img_lst, "cat b");
  // showAllImages(cat_d_img_lst, "cat d");
  saveAllImages(cat_b_img_lst, "../result_img/problem1/power_law/", "cat_bright");
  saveAllImages(cat_d_img_lst, "../result_img/problem1/power_law/", "cat_dark");
  cv::waitKey(0);
  return 0;
}
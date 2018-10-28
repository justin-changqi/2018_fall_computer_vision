#include "subtraction_operation.hpp"

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

cv::Mat sharpingImg(cv::Mat &src, cv::Mat &src_lpf, double c)
{
  cv::Mat out_img(src.rows, src.cols, CV_8UC1);
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      int px = 0;
      px = src.at<uint8_t>(i, j) - c * src_lpf.at<uint8_t>(i, j);
      px = std::max(0, px);
      px = std::min(255, px);
      out_img.at<uint8_t>(i, j) = px;
    }
  }
  return out_img;
}

cv::Mat Normalize(cv::Mat &src)
{
  cv::Mat nor_mat(src.rows, src.cols, CV_8UC1);
  std::vector<int> px_list;
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      px_list.push_back(src.at<uint8_t>(i, j));
    }
  }
  int max_px = *max_element(px_list.begin(), px_list.end());
  int min_px = *min_element(px_list.begin(), px_list.end());
  // std::cout<<"Max value: "<< max_px << std::endl;
  // std::cout<<"Min value: "<< min_px << std::endl;
  double scale = 255.0 / (max_px - min_px);
  // std::cout << scale << std::endl;
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
       nor_mat.at<uint8_t>(i, j) = src.at<uint8_t>(i, j) * scale;
    }
  }
  // std::cout << nor_mat << std::endl;
  return nor_mat;
}

int main(int argc, char **argv)
{
  cv::Mat src(512, 512, CV_8UC1);
  cv::Mat src_n(512, 512, CV_8UC1);
  loadRawFile(src, "../images/lena512.raw", 512, 512);
  loadRawFile(src_n, "../images/lena512_noise.raw", 512, 512);
  cv::Mat mask = (cv::Mat_<double>(11,11));
  setBoxMask(mask);
  cv::Mat box_img = averagingFilter(src, mask);
  cv::Mat box_n_img = averagingFilter(src_n, mask);
  cv::Mat sharped_img_03 = sharpingImg(src, box_img, 0.3);
  cv::Mat sharped_nor_img_03 = Normalize(sharped_img_03);
  cv::Mat sharped_img_05 = sharpingImg(src, box_img, 0.5);
  cv::Mat sharped_nor_img_05 = Normalize(sharped_img_05);
  cv::Mat sharped_img_08 = sharpingImg(src, box_img, 0.8);
  cv::Mat sharped_nor_img_08 = Normalize(sharped_img_08);
  cv::Mat sharped_img_1 = sharpingImg(src, box_img, 1);
  cv::Mat sharped_nor_img_1 = Normalize(sharped_img_1);
  cv::Mat sharped_img_n_03 = sharpingImg(src_n, box_n_img, 0.3);
  cv::Mat sharped_nor_n_img_03 = Normalize(sharped_img_n_03);
  showImage("src", src);
  showImage("sharped 0.3", sharped_nor_img_03);
  showImage("sharped 0.5", sharped_nor_img_05);
  showImage("sharped 0.8", sharped_nor_img_08);
  showImage("sharped 1", sharped_nor_img_1);
  showImage("sharped noise", sharped_nor_n_img_03);
  saveImage(sharped_nor_img_03, "../result_img/", "sharped_img_03");
  saveImage(sharped_nor_img_05, "../result_img/", "sharped_img_05");
  saveImage(sharped_nor_img_08, "../result_img/", "sharped_img_08");
  saveImage(sharped_nor_img_1, "../result_img/", "sharped_img_1");
  saveImage(sharped_nor_n_img_03, "../result_img/", "sharped_n_img_03");
  cv::waitKey(0);
  return 0;
}
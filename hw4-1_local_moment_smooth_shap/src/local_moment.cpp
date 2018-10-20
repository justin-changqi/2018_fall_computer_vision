#include "local_moment.hpp"

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

LocalMoment::LocalMoment(cv::Mat &src_img, int mask_size[2] )
{
  this->mask_size[0] = mask_size[0];  // x
  this->mask_size[1] = mask_size[1];  // y
  this->src_img = src_img.clone();
  this->pad_img = this->addPadding(this->src_img, mask_size);
}

cv::Mat LocalMoment::addPadding(cv::Mat &src_img, int mask_size[2])
{
  this->pad_x = (mask_size[0]-1)/2;
  this->pad_y = (mask_size[1]-1)/2;
  int pad_x = this->pad_x;
  int pad_y = this->pad_x;
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

cv::Mat LocalMoment::getLocalMean()
{
  cv::Mat img_out(240, 360, CV_8UC1);
  cv::Mat src = this->src_img;
  cv::Mat pad = this->pad_img;
  int *mask = this->mask_size;
  int pad_x = this->pad_x;
  int pad_y = this->pad_y;
  for (int i = pad_y; i < pad.rows - pad_y; i++)
  {
    for (int j = pad_x; j < pad.cols - pad_x; j++ )
    {
      double pixel_value = 0;
      for (int k = i - pad_y; k <= i + pad_y; k++)
      {
        for (int l = j - pad_x; l <= j + pad_x; l++)
        {
          pixel_value += pad.at<uint8_t>(k, l);
        }
      }
      img_out.at<uint8_t>(i-pad_y, j-pad_x) = (int)(pixel_value / (mask[0]*mask[1]));
    }
  }
  return img_out;
}

cv::Mat LocalMoment::getLocalMoment()
{

}

int main(int argc, char **argv)
{
  cv::Mat src(240, 360, CV_8UC1);
  loadRawFile(src, "../images/car.raw", 240, 360);
  int mask_size[] = {3, 3};
  LocalMoment local_moment(src, mask_size );
  // cv::Mat add_pad = local_moment.addPadding(src, mask_size);
  cv::Mat local_mean_img = local_moment.getLocalMean();
  showImage("car raw", src);
  showImage("car local mean", local_mean_img);
  // saveImage(src, "../result_img/problem1/", "p1_src");
  cv::waitKey(0);
  return 0;
}
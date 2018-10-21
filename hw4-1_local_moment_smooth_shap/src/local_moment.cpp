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

std::vector<cv::Mat> LocalMoment::getLocalMomtEnh(double E, double k0, double k1, double k2)
{
  std::vector<cv::Mat> list;
  cv::Mat img_mean(240, 360, CV_8UC1);
  cv::Mat img_var(240, 360, CV_8UC1);
  cv::Mat img_enhan(240, 360, CV_8UC1);
  cv::Mat pad = this->pad_img;
  cv::Mat src = this->src_img;
  int *mask = this->mask_size;
  int pad_x = this->pad_x;
  int pad_y = this->pad_y;
  double mean_sd_g[2];
  this->getImageMeanSd(src, mean_sd_g);
  std::cout << mean_sd_g[0] << ", " << mean_sd_g[1] << std::endl;
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
      double mean = (pixel_value / (mask[0]*mask[1]));
      double sd = sqrt(pow(pad.at<uint8_t>(i, j)-mean, 2) / (mask[0]*mask[1]));
      // for local Moment enhancement
      double enhancement;
      // std::cout << sd << std::endl;
      if (mean <= k0*mean_sd_g[0] && k1*mean_sd_g[1] <= sd && sd <= k2*mean_sd_g[1])
      {
        enhancement = E * pad.at<uint8_t>(i, j);
      } 
      else
      {
        enhancement =  pad.at<uint8_t>(i, j);
        // enhancement = 0;
      }
      img_mean.at<uint8_t>(i-pad_y, j-pad_x) = mean;
      img_var.at<uint8_t>(i-pad_y, j-pad_x) = sd;
      img_enhan.at<uint8_t>(i-pad_y, j-pad_x) = enhancement;
    }
  }
  list.push_back(img_mean);
  list.push_back(img_var);
  list.push_back(img_enhan);
  return list;
}

void LocalMoment::getImageMeanSd(cv::Mat &src_img, double *mean_sd)
{
  mean_sd[0] = 0;
  mean_sd[1] = 0;
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++ )
    {
      mean_sd[0] += src_img.at<uint8_t>(i, j);
    }
  }
  mean_sd[0] /= (src_img.rows*src_img.cols);
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++ )
    {
      double src_px = src_img.at<uint8_t>(i, j);
      mean_sd[1] += pow(src_px-mean_sd[0], 2);
    }
  }
  mean_sd[1] /= (src_img.rows*src_img.cols);
  mean_sd[1] = sqrt(mean_sd[1]);
}

int main(int argc, char **argv)
{
  cv::Mat src(240, 360, CV_8UC1);
  loadRawFile(src, "../images/car.raw", 240, 360);
  int mask_size[] = {10, 10};
  LocalMoment local_moment(src, mask_size );
  // cv::Mat add_pad = local_moment.addPadding(src, mask_size);
  std::vector<cv::Mat> lo_mean_var = local_moment.getLocalMomtEnh(4.0, 0.4, 0.02, 0.4);
  showImage("car raw", src);
  showImage("car local mean", lo_mean_var[0]);
  showImage("car local variance", lo_mean_var[1]);
  showImage("car local enhancement", lo_mean_var[2]);

  // saveImage(src, "../result_img/problem1/", "p1_src");
  cv::waitKey(0);
  return 0;
}
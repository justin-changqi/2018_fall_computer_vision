#include "smoothing_mask.hpp"

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

cv::Mat thresholding(cv::Mat &src, double th)
{
  cv::Mat out_img(src.rows, src.cols, CV_8UC1);
  std::vector<int> his(256, 0);
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      his[src.at<uint8_t>(i, j)] += 1;
    }
  }

  // get the best threshold
  int best_th = 0;
  double cdf = 0;
  double img_size = src.rows * src.cols;
  for (int i = 0; i < his.size(); i++)
  {
    cdf += his[i];
    if (cdf/img_size > th)
    {
      best_th = i;
      break;
    }
  }
  // apply threshold to image
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      if (src.at<uint8_t>(i, j) >= best_th)
      {
        out_img.at<uint8_t>(i, j) = 255;
      } 
      else
      {
        out_img.at<uint8_t>(i, j) = 0;
      }
    }
  }
  return out_img;
}

cv::Mat applyMasks(cv::Mat &src, std::vector<cv::Mat> &masks)
{
  cv::Mat out_img(src.rows, src.cols, CV_8UC1);
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      bool keep_f = true;
      for (int k = 0; k < masks.size(); k++)
      {
        if (masks[k].at<uint8_t>(i, j) == 0)
        {
          keep_f = false;
          break;
        }
      }
      if (keep_f) out_img.at<uint8_t>(i, j) = src.at<uint8_t>(i, j);
    }
  }
  return out_img;
}

int main(int argc, char **argv)
{
  cv::Mat src(368, 615, CV_8UC1);
  cv::Mat lr_mask(368, 615, CV_8UC1);
  loadRawFile(src, "../images/sheep.raw", 368, 615);
  loadRawFile(lr_mask, "../images/LRcorner.raw", 368, 615);
  cv::Mat mask = (cv::Mat_<double>(11,11));
  setBoxMask(mask);
  cv::Mat box_img = averagingFilter(src, mask);
  cv::Mat median_img = medianFilter(src, mask);
  cv::Mat th_box_filter = thresholding(box_img, 0.96);
  cv::Mat th_median_filter = thresholding(median_img, 0.96);
  // LRcorner + 2.a mask
  std::vector<cv::Mat> masks;
  masks.push_back(th_median_filter);
  masks.push_back(lr_mask);
  cv::Mat sheep_lr_mask = applyMasks(src, masks);
  showImage("sheep src", src);
  showImage("sheep LPF", box_img);
  showImage("sheep LPF th", th_box_filter);
  showImage("sheep Median", median_img);
  showImage("sheep Median th", th_median_filter);
  showImage("sheep LR corner", sheep_lr_mask);
  saveImage(src, "../result_img/problem2/", "sheep");
  saveImage(box_img, "../result_img/problem2/", "box_filter");
  saveImage(median_img, "../result_img/problem2/", "median_filter");
  saveImage(th_box_filter, "../result_img/problem2/", "box_filter_th");
  saveImage(th_median_filter, "../result_img/problem2/", "th_median_filter_th");
  saveImage(sheep_lr_mask, "../result_img/problem2/", "sheep_lr_mask");
  cv::waitKey(0);
  return 0;
}
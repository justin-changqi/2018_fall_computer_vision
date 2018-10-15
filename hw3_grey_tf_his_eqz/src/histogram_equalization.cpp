#include "histogram_equalization.hpp"

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

void writeCSV( std::string folder, 
               std::string file_name,  
               std::vector<int> curvesPoints)
{
  std::ofstream myfile(folder+file_name+".csv");
  myfile << "x,y" << std::endl;
  for (int i = 0; i < curvesPoints.size(); i++)
  {
    myfile << i << "," << curvesPoints[i] << std::endl;
  }
  myfile.close();
}

void saveImage(cv::Mat &img, std::string folder, std::string file_name)
{
  std::string save_file = folder + file_name + ".png";
  cv::imwrite(save_file, img);
}

HistogramEq::HistogramEq(cv::Mat &src_img, int L=256)
{
  this->histogram = this->getHistogram(src_img, L);
  this->cdf = this->getCDF();
  this->L = L;
  this->pixel_num = src_img.cols * src_img.rows;
  this->ComputeVmap(this->v_map);
}

std::vector<int> HistogramEq::getHistogram(cv::Mat &src_img, int L=256)
{
  std::vector<int> his(L, 0.0);
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++)
    {
      his[src_img.at<uint8_t>(i, j)] += 1.0;
    }
  }
  return his;
}

std::vector<int> HistogramEq::getCDF()
{
  std::vector<int> his_src = this->histogram;
  std::vector<int> cdf(his_src.size());
  int cdf_count = his_src[0];
  int cdf_min = 0;
  for (int i = 0; i < his_src.size(); i++)
  {
    cdf[i] = cdf_count;
    if (cdf_min == 0 && cdf_count != 0) cdf_min = cdf_count;
    cdf_count += his_src[i];
  }
  this->cdf_min = cdf_min;
  return cdf;
}

int HistogramEq::getHv(int v)
{
  return ((double)(this->cdf[v] - this->cdf_min) / (double)(this->pixel_num)) * (this->L - 1); 
}

std::vector<int> HistogramEq::getEqHistofram()
{
  std::vector<int> eq_his(this->histogram.size());
  for (int i = 0; i < eq_his.size(); i++)
  {
    eq_his[this->getHv(i)] = this->histogram[i];
  }
  return eq_his;
}

void HistogramEq::ComputeVmap(std::vector<int> &v_map)
{
  v_map.resize(this->histogram.size());
  for (int i = 0; i < this->histogram.size(); i++)
  {
    v_map[i] = this->getHv(i);
  }
}

cv::Mat HistogramEq::getEqImage(cv::Mat &img_src)
{
  cv::Mat eq_img(img_src.rows, img_src.cols, CV_8UC1);
  for (int i = 0; i < img_src.rows; i++)
  {
    for (int j = 0; j < img_src.cols; j++)
    {
      eq_img.at<char>(i, j) = this->v_map[img_src.at<uint8_t>(i, j)];
    }
  }
  return eq_img;
}

int main(int argc, char **argv)
{
  cv::Mat lvroom_b_src(512, 512, CV_8UC1);
  cv::Mat lvroom_d_src(512, 512, CV_8UC1);
  loadRawFile(lvroom_b_src, "../images/livingroom_bright.raw", 512, 512);
  loadRawFile(lvroom_d_src, "../images/livingroom_dark.raw", 512, 512);
  HistogramEq hiseq_living_b = HistogramEq(lvroom_b_src);
  HistogramEq hiseq_living_d = HistogramEq(lvroom_d_src);
  writeCSV("../result_plot_data/", "livingRoomBrightHis", hiseq_living_b.histogram);
  writeCSV("../result_plot_data/", "livingRoomDarkHis", hiseq_living_d.histogram);
  writeCSV("../result_plot_data/", "livingRoomBrightEqHis", hiseq_living_b.getEqHistofram());
  writeCSV("../result_plot_data/", "livingRoomDarkEqHis", hiseq_living_d.getEqHistofram());
  cv::Mat lvroom_b_eq_img = hiseq_living_b.getEqImage(lvroom_b_src);
  cv::Mat lvroom_d_eq_img = hiseq_living_d.getEqImage(lvroom_d_src);
  saveImage(lvroom_b_src, "../result_img/problem2/", "livingroom_bright_src");
  saveImage(lvroom_b_eq_img, "../result_img/problem2/", "livingroom_eq_bright_src");
  saveImage(lvroom_d_src, "../result_img/problem2/", "livingroom_dark_src");
  saveImage(lvroom_d_eq_img, "../result_img/problem2/", "livingroom_eq_dark_src");
  showImage("livingroom bright", lvroom_b_src);
  showImage("livingroom eq bright", lvroom_b_eq_img);
  showImage("livingroom dark", lvroom_d_src);
  showImage("livingroom eq dark", lvroom_d_eq_img);
  cv::waitKey(0);
  return 0;
}
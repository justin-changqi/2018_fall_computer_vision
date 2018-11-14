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

void saveImage(cv::Mat &img, std::string folder, std::string file_name)
{
  std::string save_file = folder + file_name + ".png";
  cv::imwrite(save_file, img);
}

double powerLaw(double L, double c, double r, double gamma)
{
  return L * c * pow(r / L, gamma);
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
      double dst_value = powerLaw(L, c, src_value, gamma);
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

void plotCurves(cv::Mat &plot, std::vector<std::vector<cv::Point2f> > curvesPoints)
{
  for (int i = 0; i < curvesPoints.size(); i++)
  {
    cv::Mat curve(curvesPoints[i], true);
    curve.convertTo(curve, CV_32S); //adapt type for polylines
    polylines(plot, curve, false, cv::Scalar(255), 2, CV_AA);
  }
}

double linearFunc(uint8_t value, cv::Point2f r1s1, cv::Point2f r2s2, double L = 255)
{
  if (value < r1s1.x)
  {
    double m = r1s1.y / r1s1.x;
    return m*value;
  }
  else if(value >= r1s1.x && value < r2s2.x)
  {
    double m = (r1s1.y - r2s2.y) / (r1s1.x - r2s2.x);
    double c = r1s1.y - m * r1s1.x;
    return m * value + c;
  }
  else if(value >= r2s2.x)
  {
    double m = (L - r2s2.y) / (L - r2s2.x);
    double c = r2s2.y - m * r2s2.x;
    return m * value + c;
  }
  else
  {
    return -1;
  }
}

void piecewiseLinearTF(cv::Mat &src_img, 
                       cv::Mat &dst_img, 
                       cv::Point2f r1s1, 
                       cv::Point2f r2s2)
{
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++)
    {
      dst_img.at<char>(i, j) = linearFunc(src_img.at<char>(i, j), r1s1, r2s2);
    }
  }
}

void writeCSV( std::string folder, 
               std::string file_name,  
               std::vector<std::vector<cv::Point2f> > curvesPoints)
{
  std::ofstream myfile(folder+file_name+".csv");
  myfile << "x";
  for (int i = 0; i < curvesPoints.size(); i++)
  {
    myfile << ",y" + std::to_string(i);
  }
  myfile << std::endl;
  for (int i = 0; i < curvesPoints[0].size(); i++)
  {
    myfile << curvesPoints[0][i].x << ",";
    for (int j = 0; j < curvesPoints.size(); j++)
    {
      myfile << curvesPoints[j][i].y;
      if (curvesPoints.size()-1 != j)
      {
        myfile << ",";
      }
    }
    myfile << std::endl;
  }
  myfile.close();
}
void writeCSV( std::string folder, 
               std::string file_name,  
               std::vector<cv::Point2f> curvesPoints)
{
  std::ofstream myfile(folder+file_name+".csv");
  myfile << "x,y" << std::endl;
  for (int i = 0; i < curvesPoints.size(); i++)
  {
    myfile << curvesPoints[i].x << "," << curvesPoints[i].y << std::endl;
  }
  myfile.close();
}

int main(int argc, char **argv)
{
  cv::Mat cat_b_src(256, 256, CV_8UC1);
  cv::Mat cat_d_src(256, 256, CV_8UC1);
  loadRawFile(cat_b_src, "../images/cat_bright.raw", 256, 256);
  loadRawFile(cat_d_src, "../images/cat_dark.raw", 256, 256);
  
  // Power-Law Transformation
  std::vector<cv::Mat> cat_b_img_lst;
  std::vector<cv::Mat> cat_d_img_lst;
  std::vector<std::vector<cv::Point2f> > curvesPoints;
  for (int i = 0; i < sizeof(GAMMAS)/sizeof(double); i++)
  {
    cv::Mat cat_b_transtormed(256, 256, CV_8UC1);
    cv::Mat cat_d_transtormed(256, 256, CV_8UC1);
    PowerLawTransformation(cat_b_src, cat_b_transtormed, GAMMAS[i]);
    PowerLawTransformation(cat_d_src, cat_d_transtormed, GAMMAS[i]);
    cat_b_img_lst.push_back(cat_b_transtormed);
    cat_d_img_lst.push_back(cat_d_transtormed);
    // insert data to curve
    std::vector<cv::Point2f> curvePoints;
    for (int j = 0; j < 256; j++)
    {
      cv::Point2f point(j, powerLaw(255, 1.0, j, GAMMAS[i]));
      curvePoints.push_back(point);
    }
    curvesPoints.push_back(curvePoints);
  }
  cv::Mat plot_img(256, 256, CV_8UC1, cv::Scalar(0));
  plotCurves(plot_img, curvesPoints);

  // Piecewise-Linear Transformation
  cv::Mat cat_b_plt(256, 256, CV_8UC1);
  cv::Mat cat_d_plt(256, 256, CV_8UC1);
  piecewiseLinearTF(cat_b_src, cat_b_plt, cv::Point2f(20,10), cv::Point2f(150,50));
  piecewiseLinearTF(cat_d_src, cat_d_plt, cv::Point2f(10,150), cv::Point2f(50,200));
  std::vector<cv::Point2f> piecewise_curve_bright;
  std::vector<cv::Point2f> piecewise_curve_dark;
  for (int i = 0; i < 256; i++)
  {
    piecewise_curve_bright.push_back(cv::Point2f(i, linearFunc(i, cv::Point2f(20, 10), cv::Point2f(150,50))));
    piecewise_curve_dark.push_back(cv::Point2f(i, linearFunc(i, cv::Point2f(10, 150), cv::Point2f(50,200))));
  }

  // showImage("Power Law", plot_img);
  // showAllImages(cat_b_img_lst, "cat b");
  // showAllImages(cat_d_img_lst, "cat d");
  // showImage("src cat bright", cat_b_src);
  // showImage("PLT cat bright", cat_b_plt);
  // showImage("src cat dark", cat_d_src);
  // showImage("PLT cat dark", cat_d_plt);
  writeCSV("../result_plot_data/", "Power-Law", curvesPoints);
  writeCSV("../result_plot_data/", "piecewise_curve_bright", piecewise_curve_bright);
  writeCSV("../result_plot_data/", "piecewise_curve_dark", piecewise_curve_dark);
  saveAllImages(cat_b_img_lst, "../result_img/problem1/power_law/", "cat_bright");
  saveAllImages(cat_d_img_lst, "../result_img/problem1/power_law/", "cat_dark");
  saveImage(cat_b_src, "../result_img/problem1/", "cat_bright_src");
  saveImage(cat_b_plt, "../result_img/problem1/", "cat_bright_plt");
  saveImage(cat_d_src, "../result_img/problem1/", "cat_dark_src");
  saveImage(cat_d_plt, "../result_img/problem1/", "cat_dark_plt");
  cv::waitKey(0);
  return 0;
}
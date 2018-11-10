#include <dft_2d.hpp>

Dft2d::Dft2d(cv::Mat &src_img)
{
  this->src_img_ = src_img.clone();
}

cv::Mat Dft2d::getDftImg()
{
  static cv::Mat img_out(src_img_.rows, src_img_.cols, CV_64FC1);
  for (int i = 0; i < src_img_.rows; i++) 
  {
    for (int j = 0; j < src_img_.cols; j++) 
    {
      std::complex<double> Fxy;
      Fxy = this->getDftValue(j, i);
      img_out.at<double>(i, j) = std::abs(Fxy);
      std::cout << "\rCalculating DFT " << std::setprecision(3) 
                << ((i*src_img_.rows+j)*100.0)/(double)(src_img_.rows*src_img_.cols)
                << " %";
    }
  }
  std::cout << std::endl;
  return img_out;
}

void Dft2d::DftTask(int max_row, &cv::Mat &img)
{

}

std::complex<double> Dft2d::getDftValue(int u, int v)
{
  std::complex<double> result(0, 0);
  const double M = src_img_.cols;
  const double N = src_img_.rows;
  const std::complex<double> i(0, 1);
  for (int y = 0; y < N; y++) 
  {
    for (int x = 0; x < M; x++) 
    {
      double fxy = src_img_.at<uint8_t>(y, x);
      result += fxy*exp(-2.0*i*M_PI*(u*x/M+v*y/M));
    }
  }
  return result/(M*N);
}
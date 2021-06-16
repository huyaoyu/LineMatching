//
// Created by yaoyu on 6/1/21.
//

#ifndef LINEMATCHING_IMAGE_UTILS_HPP
#define LINEMATCHING_IMAGE_UTILS_HPP

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace iu
{
    cv::Mat read_image(const std::string& fn);
    cv::Mat resize_by_longer_edge(const cv::Mat& src, int longer_edge);
    cv::Mat grey_2_BGR(const cv::Mat& grey);
    cv::Mat binaries(const cv::Mat& img, int threshold=10);
    cv::Mat dilate_erode(const cv::Mat& img);
    cv::Mat adjust_contrast_brightness(const cv::Mat& img, double alpha, double beta);
}

#endif //LINEMATCHING_IMAGE_UTILS_HPP

//
// Created by yaoyu on 6/1/21.
//

#include "image_utils/image_utils.hpp"

namespace iu
{
    cv::Mat read_image(const std::string& fn) {
        cv::Mat img = cv::imread(fn, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::stringstream ss;
            ss << fn << " read failed. ";
            throw std::runtime_error(ss.str());
        }

        return img;
    }

    cv::Mat resize_by_longer_edge(const cv::Mat& src, int longer_edge) {
        // Get the new shape.
        int new_row, new_col;
        if ( src.rows >= src.cols ) {
            new_row = longer_edge;
            new_col = static_cast<int>(std::round(1.0f * new_row / src.rows * src.cols));
        } else {
            new_col = longer_edge;
            new_row = static_cast<int>(std::round(1.0f * new_col / src.cols * src.rows));
        }

        // Resize.
        cv::Mat dst;
        cv::resize( src, dst, cv::Size2i(new_col, new_row), 0, 0, cv::INTER_CUBIC );
        return dst;
    }

    cv::Mat grey_2_BGR(const cv::Mat& grey) {
        if ( grey.channels() != 1 ) return grey;

        cv::Mat out;
        cv::cvtColor( grey, out, cv::COLOR_GRAY2BGR );
        return out;
    }

    cv::Mat binaries(const cv::Mat& img, int threshold) {
        cv::Mat blurred;
        cv::GaussianBlur(img, blurred, cv::Size(5, 5), 0);
        cv::Mat out;
        cv::threshold(img, out, threshold, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        return out;
    }

    cv::Mat dilate_erode(const cv::Mat& img) {
        cv::Mat dilated;
        cv::dilate(img, dilated, cv::Mat());
        cv::Mat blurred;
        cv::GaussianBlur(dilated, blurred, cv::Size(3, 3), 0);
        cv::Mat eroded;
        cv::erode(blurred, eroded, cv::Mat());
        return eroded;
    }

    cv::Mat adjust_contrast_brightness(const cv::Mat& img, double alpha, double beta) {
        cv::Mat out;
        cv::convertScaleAbs(img, out, alpha, beta);
        return out;
    }
}
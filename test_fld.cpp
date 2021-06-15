//
// Created by yaoyu on 6/1/21.
//

#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/ximgproc.hpp>

#include "image_utils/image_utils.hpp"

namespace cvx = cv::ximgproc;

int main(int argc, char** argv) {
    std::cout << "Hello, test_lsd! \n";

    // The input image.
    const std::string img_fn {"../data/0839-0001-13.jpg"};

    // Read the image.
    cv::Mat img = iu::read_image(img_fn);

    // Resize.
    cv::Mat img_resized = iu::resize_by_longer_edge(img, 1024);

    // Create the detector.
    const int length_threshold = 100;
    const double distance_threshold = std::sqrt(2);
    const double canny_th1 = 50.0;
    const double canny_th2 = 50.0;
    const int canny_aperture = 3;
    const bool flag_merge = false;
    cv::Ptr<cvx::FastLineDetector> fld = cvx::createFastLineDetector(
            length_threshold, distance_threshold,
            canny_th1, canny_th2, canny_aperture,
            flag_merge);

    std::vector<cv::Vec4f> lines;

    // Detect lines.
    fld->detect( img_resized, lines );

    // Visualization.
    cv::Mat line_img(img_resized);
    fld->drawSegments( line_img, lines );

    // Write line_img
    cv::imwrite("./line_img_fld.png", line_img);

    return 0;
}

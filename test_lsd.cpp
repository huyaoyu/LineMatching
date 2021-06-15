//
// Created by yaoyu on 6/1/21.
//

#include <iostream>

#include <opencv2/opencv_modules.hpp>
#ifdef HAVE_OPENCV_FEATURES2D

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

#include "image_utils/image_utils.hpp"

namespace cvline = cv::line_descriptor;

int main(int argc, char** argv) {
    std::cout << "Hello, test_lsd! \n";

    // The input image.
    const std::string img_fn {"../data/0839-0001-13.jpg"};

    // Read the image.
    cv::Mat img = iu::read_image(img_fn);

    // Resize.
    cv::Mat img_resized = iu::resize_by_longer_edge(img, 1024);

    // Random binary mask.
    cv::Mat mask = cv::Mat::ones( img_resized.size(), CV_8UC1 );

    // Binary descriptor object.
    cv::Ptr<cvline::LSDDetector> desc =
            cvline::LSDDetector::createLSDDetector();

    // Container for the lines.
    std::vector<cvline::KeyLine> lines;

    // Line extraction.
    cv::Mat output = img_resized.clone();
    desc->detect( img_resized, lines, 2, 1, mask );

    // Draw lines for visualization.
    if ( output.channels() == 1 ) cv::cvtColor( output, output, cv::COLOR_GRAY2BGR );
    for ( const auto& kl : lines ) {
        if ( kl.octave == 0 ) {
            // Random color.
            const auto R = static_cast<int>( std::rand() * 255);
            const auto G = static_cast<int>( std::rand() * 255);
            const auto B = static_cast<int>( std::rand() * 255);

            // The end-points of the line.
            const cv::Point2f pt0 { kl.startPointX, kl.startPointY };
            const cv::Point2f pt1 { kl.endPointX, kl.endPointY };

            // Draw line.
            cv::line( output, pt0, pt1, cv::Scalar(B, G, R), 1 );
        }
    }

    // Write the image.
    cv::imwrite("./Lines.png", output);

    return 0;
}

#else

int main(int argc, char** argv) {
    std::cout << "test_lsd: OpenCV is built without features2d moudle. \n";

    return 0;
}

#endif
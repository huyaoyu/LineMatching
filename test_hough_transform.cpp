#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "image_utils/image_utils.hpp"

int main(int argc, char** argv) {
    std::cout << "Hello, LineMatching! \n";

    const std::string img_fn = "../data/0839-0001-13.jpg";

    // Read the image.
    cv::Mat img = iu::read_image(img_fn);

    // Resize.
    cv::Mat img_resized = iu::resize_by_longer_edge(img, 1024);

    // Edge detection.
    cv::Mat canny_dst;
    cv::Canny( img_resized, canny_dst, 50, 200, 3);

    // Standard Hough line transform.
    std::vector<cv::Vec2f> lines;
    cv::HoughLines( canny_dst, lines, 1, CV_PI/180, 150, 0, 0 );

    // Draw the lines.
    cv::Mat vis;
    cv::cvtColor(canny_dst, vis, cv::COLOR_GRAY2BGR);
    for ( const auto& line : lines ) {
        const float rho = line[0];
        const float theta = line[1];
        cv::Point pt1, pt2;
        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;

        pt1.x = std::round(x0 + 1000*(-b));
        pt1.y = std::round(y0 + 1000*( a));
        pt2.x = std::round(x0 - 1000*(-b));
        pt2.y = std::round(y0 - 1000*( a));

        cv::line( vis, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA );
    }

    // Probabilistic Line Transform.
    std::vector<cv::Vec4i> lines_p;
    cv::HoughLinesP(canny_dst, lines_p, 1, CV_PI/180, 50, 50, 10);

    // Draw the lines.
    cv::Mat vis_p;
    cv::cvtColor(canny_dst, vis_p, cv::COLOR_GRAY2BGR);
    for ( const auto& line : lines ) {
        cv::line( vis_p,
                  cv::Point( line[0], line[1] ),
                  cv::Point( line[2], line[3] ),
                  cv::Scalar(0, 0, 255),
                  3,
                  cv::LINE_AA);
    }

    // Write results.
    cv::imwrite("./standard_hough.png", vis);
    cv::imwrite("./probabilistic_hough.png", vis_p);

    return 0;
}

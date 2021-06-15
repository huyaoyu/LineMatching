//
// Created by yaoyu on 6/1/21.
//

#include <iostream>
#include <string>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

#include "image_utils/image_utils.hpp"

namespace cvline = cv::line_descriptor;

static cv::Mat read_resize(const std::string& fn, int longer_edge=1024) {
    // Read the image.
    cv::Mat img = iu::read_image(fn);

    // Resize.
    return iu::resize_by_longer_edge(img, 1024);
}

static void show_keyline_info(const std::vector<cvline::KeyLine>& keylines) {
    const int N = keylines.size();

    std::cout << N << " keylines from all octave levels. \n";

    for ( int i = 0; i < keylines.size(); i++ ) {
        std::cout << "Keyline " << i << ": \n";
        const auto& keyline = keylines[i];
        std::cout << "angle = " << keyline.angle << "\n";
        std::cout << "class_id = " << keyline.class_id << "\n";
        std::cout << "octave = " << keyline.octave << "\n";
        std::cout << "middle point: " << keyline.pt.x << ", " << keyline.pt.y << "\n";
        std::cout << "length = " << keyline.lineLength << "\n";

        const float x_diff = keyline.startPointX - keyline.endPointX;
        const float y_diff = keyline.startPointY - keyline.endPointY;
        const float length_xy = std::sqrt( x_diff * x_diff + y_diff * y_diff );
        std::cout << "length_xy = " << length_xy << "\n";
    }
}

static void filter_keylines(
        const std::vector<cvline::KeyLine>& keylines,
        const cv::Mat& line_desc,
        std::vector<cvline::KeyLine>& filtered_keylines,
        cv::Mat& filtered_line_desc,
        int octave) {
    for ( int i = 0; i < keylines.size(); i++ ) {
        if ( keylines[i].octave == octave ) {
            filtered_keylines.push_back( keylines[i] );
            filtered_line_desc.push_back( line_desc.row(i) );
        }
    }
}

static void filter_keylines(
        const std::vector<cvline::KeyLine>& keylines,
        const cv::Mat& line_desc,
        std::vector<cvline::KeyLine>& filtered_keylines,
        cv::Mat& filtered_line_desc,
        int octave,
        float scale,
        float length) {
    const float scale_octave = std::pow( scale, octave );

    for ( int i = 0; i < keylines.size(); i++ ) {
        const auto& keyline = keylines[i];
        if ( keyline.octave == octave ) {
            if ( keyline.lineLength * scale_octave >= length ) {
                filtered_keylines.push_back( keyline );
                filtered_line_desc.push_back( line_desc.row(i) );
            }
        }
    }
}

static cv::Vec2f convert_keyline_2_vector(
        const cvline::KeyLine& keyline) {
    cv::Vec3f vec;

    vec = cv::Vec3f( keyline.startPointX, keyline.startPointY, 1 ).cross(
            cv::Vec3f( keyline.endPointX, keyline.endPointY, 1 ) );

    if ( vec[2] != 0 ) {
        vec[0] /= vec[2];
        vec[1] /= vec[2];
        vec[2]  = 1.f;
    }

    cv::Vec2f vec2;
    vec2[0] = vec[0];
    vec2[1] = vec[1];

    return vec2;
}

static void collect_and_convert_keylines(
        const std::vector<cvline::KeyLine>& keylines_0,
        const std::vector<cvline::KeyLine>& keylines_1,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Vec2f>& vec_0,
        std::vector<cv::Vec2f>& vec_1) {
    // Clear vec_0 and vec_1.
    const int N = matches.size();
    vec_0.resize(N);
    vec_1.resize(N);

    // Convert the KeyLine objects.
    for ( int i = 0; i < N; i++ ) {
        const auto& m = matches[i];
        vec_0[i] = convert_keyline_2_vector(keylines_0[m.trainIdx]);
        vec_1[i] = convert_keyline_2_vector(keylines_1[m.queryIdx]);
    }
}

template < typename VecT >
static cv::Mat vecf_array_2_cv_mat(
        const std::vector<VecT>& vecs) {
    const auto D = static_cast<int>( VecT::channels );
    const int N = vecs.size();

    int type;
    switch (D) {
        case 1:
        {
            type = CV_32FC1;
            break;
        }
        case 2:
        {
            type = CV_32FC2;
            break;
        }
        case 3:
        {
            type = CV_32FC3;
            break;
        }
        default:
        {
            type = CV_32FC3;
            break;
        }
    }

    cv::Mat m(N, 1, type);

    for ( int i = 0; i < N; i++ ) {
        m.at<VecT>(i, 0) = vecs[i];
    }

    return m;
}

static cv::Mat compute_homograpy_by_lines(
        const std::vector<cvline::KeyLine>& lines_0,
        const std::vector<cvline::KeyLine>& lines_1,
        const std::vector<cv::DMatch>& matches) {

    // Convert the KeyLine objects into vector representations.
    std::vector<cv::Vec2f> vec_0, vec_1;
    collect_and_convert_keylines( lines_0, lines_1, matches, vec_0, vec_1 );

    // The A matrix.
    cv::Mat A( 0, 9, CV_32FC1 );
    const int N = matches.size();
    for ( int i = 0; i < N; i++ ) {
        const float x = vec_0[i][0]; // Train.
        const float y = vec_0[i][1];
        const float u = vec_1[i][0]; // Query.
        const float v = vec_1[i][1];

        std::cout << "xyuv: "
                  << x << ", " << y << ", "
                  << u << ", " << v << "\n";

        cv::Mat a = cv::Mat::zeros(2, 9, CV_32FC1);

        a.at<float>(0, 0) = -u;
        a.at<float>(0, 2) =  u * x;
        a.at<float>(0, 3) = -v;
        a.at<float>(0, 5) =  v * x;
        a.at<float>(0, 6) = -1.f;
        a.at<float>(0, 8) =  x;

        a.at<float>(1, 1) = -u;
        a.at<float>(1, 2) =  u * y;
        a.at<float>(1, 4) = -v;
        a.at<float>(1, 5) =  v * y;
        a.at<float>(1, 7) = -1.f;
        a.at<float>(1, 8) =  y;

        cv::vconcat( A, a, A );
    }

    // SVD.
    cv::SVD svd(A);
    cv::Mat right_singular;
    cv::transpose(svd.vt, right_singular);
    cv::Mat h = right_singular.col( right_singular.cols - 1 );
    std::cout << "h = " << h << "\n";

    // The clone() is necessary because h is not continuous.
    cv::Mat hg = h.clone().reshape(1, 3);
    hg = hg / hg.at<float>(2,2);

    // Do some check.
    cv::Mat line_vec_0(3, 1, CV_32FC1);
    cv::Mat line_vec_1(3, 1, CV_32FC1);
    line_vec_0.at<float>(0, 0) = vec_0[0][0];
    line_vec_0.at<float>(1, 0) = vec_0[0][1];
    line_vec_0.at<float>(2, 0) = vec_0[0][2];

    line_vec_1.at<float>(0, 0) = vec_1[0][0];
    line_vec_1.at<float>(1, 0) = vec_1[0][1];
    line_vec_1.at<float>(2, 0) = vec_1[0][2];

    cv::Mat line_vec_1_t = hg.t() * line_vec_1;

    std::cout << "line_vec_1_t = \n" << line_vec_1_t << "\n";
    std::cout << "line_vec_0 = \n" << line_vec_0 << "\n";

    return hg;
}

static void test_binary_detector(
        const cv::Mat& img0,
        const cv::Mat& img1) {
    // Create the masks.
    cv::Mat mask0 = cv::Mat::ones( img0.size(), CV_8UC1 );
    cv::Mat mask1 = cv::Mat::ones( img0.size(), CV_8UC1 );

    // Binary descriptor.
    cv::Ptr<cvline::BinaryDescriptor> desc = cvline::BinaryDescriptor::createBinaryDescriptor();

    // Compute the lines and descriptors.
    std::vector<cvline::KeyLine> keyline0, keyline1;
    cv::Mat desc0, desc1;

    (*desc)( img0, mask0, keyline0, desc0, false, false );
    (*desc)( img1, mask1, keyline1, desc1, false, false );

    // Filter keylines.
    std::vector<cvline::KeyLine> filtered_lines_0, filtered_lines_1;
    cv::Mat filtered_line_desc_0, filtered_line_desc_1;
    filter_keylines( keyline0, desc0, filtered_lines_0, filtered_line_desc_0, 0 );
    filter_keylines( keyline1, desc1, filtered_lines_1, filtered_line_desc_1, 0 );

    // Binary descriptor matcher.
    cv::Ptr<cvline::BinaryDescriptorMatcher> matcher =
            cvline::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    // Match.
    std::vector<cv::DMatch> matches;
    matcher->match( filtered_line_desc_1, filtered_line_desc_0, matches );

    // Select the good matches.
    std::vector<cv::DMatch> good_matches;
    for ( auto& m : matches ) {
        if ( m.distance < 25) good_matches.push_back( m );
    }

    // Draw the line matches.
    cv::Mat out_img;
    std::vector<char> mask( matches.size(), 1 );
    cvline::drawLineMatches( img0, filtered_lines_0,
                             img1, filtered_lines_1,
                             good_matches,
                             out_img,
                             cv::Scalar::all(-1),
                             cv::Scalar::all(-1),
                             mask,
                             cvline::DrawLinesMatchesFlags::DEFAULT);
    // Write the image.
    cv::imwrite("./LineMatching.png", out_img);
}

static void test_lsd(
        const cv::Mat& img0,
        const cv::Mat& img1) {
    // Create the masks.
    cv::Mat mask0 = cv::Mat::ones( img0.size(), CV_8UC1 );
    cv::Mat mask1 = cv::Mat::ones( img0.size(), CV_8UC1 );

    // LSD detector.
    cv::Ptr<cvline::LSDDetector> lsd = cvline::LSDDetector::createLSDDetector();

    // Detect lines.
    std::vector<cvline::KeyLine> keylines0, keylines1;
    cv::Mat desc0, desc1;
    lsd->detect( img0, keylines0, 2, 2, mask0 );
    lsd->detect( img1, keylines1, 2, 2, mask1 );

    // Show the info of these key lines.
    std::cout << "keyliens0: \n";
    show_keyline_info( keylines0 );
    std::cout << "\nkeyliens1: \n";
    show_keyline_info( keylines1 );

    // Binary descriptor.
    cv::Ptr<cvline::BinaryDescriptor> desc = cvline::BinaryDescriptor::createBinaryDescriptor();

    // Compute descriptors from the first octave.
    desc->compute( img0, keylines0, desc0 );
    desc->compute( img1, keylines1, desc1 );

    // Filter the lines and descriptors.
    std::vector<cvline::KeyLine> filtered_lines_0, filtered_lines_1;
    cv::Mat filtered_desc_0, filtered_desc_1;
    filter_keylines( keylines0, desc0, filtered_lines_0, filtered_desc_0, 0, 2, 50.f );
    filter_keylines( keylines1, desc1, filtered_lines_1, filtered_desc_1, 0, 2, 50.f );

    // Match.
    // Binary descriptor matcher.
    cv::Ptr<cvline::BinaryDescriptorMatcher> matcher =
            cvline::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    std::vector<cv::DMatch> matches;
    // Query & train.
    matcher->match( filtered_desc_1, filtered_desc_0, matches );

    std::vector<cv::DMatch> good_matches;
    for ( auto& m : matches ) {
        if ( m.distance < 25 ) good_matches.push_back(m);
    }

    std::cout << "good_matches.size() = " << good_matches.size() << "\n";

    // Homography.
    cv::Mat hg = compute_homograpy_by_lines(
            filtered_lines_0, filtered_lines_1, good_matches);

    std::cout << "hg = \n" << hg << "\n";

    cv::Mat out_img;
    std::vector<char> mask( matches.size(), 1 );

    // Convert the gray image into BGR.
    cv::Mat img0_bgr = iu::grey_2_BGR(img0);
    cv::Mat img1_bgr = iu::grey_2_BGR(img1);

    cvline::drawLineMatches(
            img0_bgr, filtered_lines_0,
            img1_bgr, filtered_lines_1,
            good_matches,
            out_img,
            cv::Scalar::all(-1),
            cv::Scalar::all(-1),
            mask,
            cvline::DrawLinesMatchesFlags::DEFAULT);

    // Write.
    cv::imwrite("LineMatching_LSD.png", out_img);

}

int main(int argc, char** argv) {
    std::cout << "Hello, test_lsd_matching! \n";

    // Read and resize the images.
//    cv::Mat img0 = read_resize("../data/0839-0001-05.jpg");
//    cv::Mat img1 = read_resize("../data/0839-0002-05.jpg");
//    cv::Mat img0 = read_resize("../data/0839-0001-08.jpg");
//    cv::Mat img1 = read_resize("../data/0839-0002-08.jpg");
    cv::Mat img0 = read_resize("../data/0839-0001-13.jpg");
    cv::Mat img1 = read_resize("../data/0839-0002-13.jpg");

//    // Binaries.
//    img0 = iu::binaries(img0);
//    img1 = iu::binaries(img1);
//
//    // Dilate-erode.
//    img0 = iu::dilate_erode(img0);
//    img1 = iu::dilate_erode(img1);

    // Binary descriptor.
    // test_binary_detector(img0, img1);

    // LSD.
    test_lsd(img0, img1);

    return 0;
}
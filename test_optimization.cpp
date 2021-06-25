//
// Created by yaoyu on 6/1/21.
//

// ========== C++ standard headers. ==========
#include <iostream>
#include <limits>
#include <string>

// ========== Third-party headers. ==========
// Boost.
#include <boost/math/constants/constants.hpp>

// Ceres.
#include <ceres/ceres.h>
#include <glog/logging.h>

// OpenCV.
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

// ========== Project headers. ==========
#include "simple_timer.hpp"
#include "image_utils/image_utils.hpp"

// ========== Name space alias. ==========
namespace cvline = cv::line_descriptor;

// Globals.
const auto PI = boost::math::constants::pi<double>();
const auto TEN_DEGREE_RAD = 10.0 / 180 * PI;
const auto MAX_D = std::numeric_limits<double>::max();

const int NO_RESIZE = -1;
const double NO_WEIGHT_SIGMA = -1.;

/**
 * Read and resize an image. The resize is done by scale the longer edge to the value
 * specified by \p longer_edge.
 *
 * @param fn Image filename.
 * @param longer_edge Length of the resized longer edge. Use a non-positive value to disable.
 * @return resized image.
 */
static cv::Mat read_resize(const std::string& fn, int longer_edge=1024) {
    // Read the image.
    cv::Mat img = iu::read_image(fn);

    // Resize.
    if ( longer_edge > 0 ) {
        return iu::resize_by_longer_edge(img, 1024);
    } else {
        return img;
    }
}

/**
 * A convenient representation of the input data.
 */
struct MatchInput {
    MatchInput( const std::string& name,
                int resize_longer_edge=NO_RESIZE,
                int keyline_filter=10,
                double weight_sigma=100.)
            : name{name}
            , resize_longer_edge{resize_longer_edge}
            , keyline_filter{keyline_filter}
            , weight_sigma{weight_sigma} { }

    void read_img0(const std::string& fn) {
        img0 = read_resize(fn, resize_longer_edge);
    }

    void read_img1(const std::string& fn) {
        img1 = read_resize(fn, resize_longer_edge);
    }

    cv::Mat img0;           // The reference/target/training image.
    cv::Mat img1;           // The test/source image.
    std::string name;       // Name of this input case.
    int resize_longer_edge; // Target longer edge length after resizing.
    int keyline_filter;     // Minimum length of a keyline.
    double weight_sigma;    // The weight coefficient of the loss function.
};

/**
 * Print some info of a collection of keylines.
 * @param keylines
 */
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

/**
 * Filter keylines by the octave and length.
 * @param keylines A collection of keylines.
 * @param filtered_keylines Output keylines.
 * @param octave Octave number.
 * @param scale Scale between octaves.
 * @param length Length threshold.
 */
static void filter_keylines(
        const std::vector<cvline::KeyLine>& keylines,
        std::vector<cvline::KeyLine>& filtered_keylines,
        int octave,
        float scale,
        float length) {
    // Scale factor of the target octave.
    const float scale_octave = std::pow( scale, octave );

    for( const auto& keyline : keylines) {
        if ( keyline.octave == octave ) {
            if ( keyline.lineLength * scale_octave >= length ) {
                filtered_keylines.push_back( keyline );
            }
        }
    }
}

/**
 * Draw keylines on an image.
 * @param img The original image.
 * @param keylines A collection of keylines.
 * @return A new image with keylines drawn.
 */
static cv::Mat draw_keylines(
        const cv::Mat img,
        const std::vector<cvline::KeyLine>& keylines) {
    // Make a copy of the input mat.
    cv::Mat img_lines = iu::grey_2_BGR(img);

    // Loop over all the key lines.
    for ( const auto& kl : keylines ) {
        cv::line(img_lines,
                 cv::Point( static_cast<int>(kl.startPointX), static_cast<int>(kl.startPointY) ),
                 cv::Point( static_cast<int>(kl.endPointX), static_cast<int>(kl.endPointY) ),
                 cv::Scalar( 0, 255, 0 ),
                 2,
                 cv::LINE_AA);
    }

    return img_lines;
}

/**
 * Draw keylines on both of the reference and test images.
 * @param mi An object contains the images.
 * @param key_lines_0 Keylines on the reference image.
 * @param key_lines_1 Keylines on the test image.
 */
static void draw_keyline_mi(
        const MatchInput& mi,
        const std::vector<cvline::KeyLine>& key_lines_0,
        const std::vector<cvline::KeyLine>& key_lines_1) {
    cv::Mat img0_lines = draw_keylines(mi.img0, key_lines_0);
    std::stringstream ss;
    ss << "./" << mi.name << "_img0_lines.png";
    cv::imwrite( ss.str(), img0_lines );

    cv::Mat img1_lines = draw_keylines(mi.img1, key_lines_1);
    ss.str(""); ss.clear();
    ss << "./" << mi.name << "_img1_lines.png";
    cv::imwrite( ss.str(), img1_lines );
}

/**
 * A representation of a 2D line.
 */
struct SimpleLine {
    SimpleLine(double px0, double py0, double px1, double py1)
    : px0{px0}, py0{py0}, px1{px1}, py1{py1}
    {
        c = px0 * py1 - py0 * px1;

        if ( c != 0 ) {
            a = ( py0 - py1 ) / c;
            b = ( px1 - px0 ) / c;
            c = 1;
        } else {
            a = py0 - py1;
            b = px1 - px0;
        }

        sqrt_a2b2 = std::sqrt( a * a + b * b );

        const double dx = px1 - px0;
        const double dy = py1 - py0;

        angle = std::atan2( dy, dx );
        angle = angle < 0 ? angle + PI : angle;

        length = std::sqrt( dx * dx + dy * dy );
    }

    /**
     * Compute the distance from a point to the current line.
     * @param px x-coordinate.
     * @param py y-coordinate.
     * @return distacne.
     */
    [[nodiscard]] double point_dist( double px, double py ) const {
        return std::abs( a * px + b * py + c ) / sqrt_a2b2;;
    }

    /**
     * Return the middle point coordinate.
     * @return x, y coordinate.
     */
    [[nodiscard]] std::pair<double, double> middle() const {
        return { ( px0 + px1 ) / 2, ( py0 + py1 ) / 2 };
    }

    double px0; // x-coordinate of the first point.
    double py0; // y-coordinate of the first point.
    double px1; // x-coordinate of the second point.
    double py1; // y-coordinate of the second point.
    double a;   // Line coefficients.
    double b;
    double c;
    double sqrt_a2b2; // Intermediate value.
    double angle;     // Angle with respect to the x-axis. Radian.
    double length;    // Length of the line.
};

/**
 * A line representation for Ceres, similar to \ref SimpleLine.
 * @tparam rT Data type determined by Ceres.
 */
template < typename rT >
struct SimpleLineCeres {
    SimpleLineCeres(const rT& px0, const rT& py0, const rT& px1, const rT& py1)
            : px0{px0}, py0{py0}, px1{px1}, py1{py1}
    {
        c = px0 * py1 - py0 * px1;

        if ( c != 0.0 ) {
            a = ( py0 - py1 ) / c;
            b = ( px1 - px0 ) / c;
            c = rT(1.0);
        } else {
            a = py0 - py1;
            b = px1 - px0;
        }

        sqrt_a2b2 = ceres::sqrt( a * a + b * b );

        const rT dx = px1 - px0;
        const rT dy = py1 - py0;

        angle = ceres::atan2( dy, dx );
        angle = angle < 0.0 ? angle + PI : angle;

        length = ceres::sqrt( dx * dx + dy * dy );
    }

    rT point_dist( const rT& px, const rT& py ) const {
        return ceres::abs( a * px + b * py + c ) / sqrt_a2b2;;
    }

    rT px0;
    rT py0;
    rT px1;
    rT py1;
    rT a;
    rT b;
    rT c;
    rT sqrt_a2b2;
    rT angle;
    rT length;
};

/**
 * Convert a set of keylines to SimpleLine objects.
 * @param key_lines Input keylines.
 * @return SimpleLine objects.
 */
static std::vector<SimpleLine> collect_and_convert_keylines(
        const std::vector<cvline::KeyLine>& key_lines) {
    std::vector<SimpleLine> simple_lines;

    // Convert the KeyLine objects.
    for ( const auto& key_line : key_lines ) {
        simple_lines.emplace_back(
                key_line.startPointX, key_line.startPointY,
                key_line.endPointX, key_line.endPointY);
    }

    return simple_lines;
}

/**
 * Cost function for Ceres.
 */
struct CF_SingleLine{
    CF_SingleLine(
            const std::vector<SimpleLine>* ref_lines,
            const SimpleLine* tst_line,
            double max_dist,
            double weight_sigma,
            int* best_match_index= nullptr)
            : ref_lines(ref_lines), tst_line(tst_line)
            , max_dist(max_dist), weight_sigma(weight_sigma)
            , best_match_index(best_match_index) {}

    /**
     * Homography projection. \p h contains the 9 elements of a homography matrix, in
     * row major order. \p px and \p py are the pixel coordinated before transformation.
     * \p ppx and \p ppy are the transformed values.
     *
     * @tparam rT Value type of all the floating point numbers.
     * @param h Homography matrix.
     * @param px x-coordinate before transformation.
     * @param py y-coordinate before transformation.
     * @param ppx x-coordinate after transformation.
     * @param ppy y-coordinate after transformation.
     */
    template < typename rT >
    void project(
            const rT* const h, const rT& px, const rT& py,
            rT& ppx, rT& ppy) const {
        ppx = h[0] * px + h[1] * py + h[2];
        ppy = h[3] * px + h[4] * py + h[5];
        const auto ppz = h[6] * px + h[7] * py + h[8];

        // Homogenous coordinates.
        if ( ppz != 0.0 ) {
            ppx /= ppz;
            ppy /= ppz;
        }
    }

    /**
     * Compute the residual for the optimization.
     * @tparam rT Data type determined by Ceres.
     * @param h Optimization target.
     * @param residual Residual
     * @return Always true.
     */
    template < typename rT >
    bool operator () ( const rT* const h, rT* residual ) const {
        // Project the end points.
        rT ppx0, ppy0, ppx1, ppy1;
        project( h, rT(tst_line->px0), rT(tst_line->py0), ppx0, ppy0 );
        project( h, rT(tst_line->px1), rT(tst_line->py1), ppx1, ppy1 );

        // Create a new SimpleLineCeres object.
        SimpleLineCeres<rT> slc( ppx0, ppy0, ppx1, ppy1 );

        // Find the nearest line.
        rT dist = rT(MAX_D);
        double ref_length = tst_line->length;
        int best_index = INVALID_BEST_MATCH_INDEX;
        for ( int i = 0; i < ref_lines->size(); i++ ) {
            const auto& line = (*ref_lines)[i];

            // Check angle.
            if ( ceres::abs( slc.angle - line.angle ) > TEN_DEGREE_RAD ) continue;

            // Compute distance.
            rT d = slc.point_dist( rT(line.px0), rT(line.py0) )
                 + slc.point_dist( rT(line.px1), rT(line.py1) );

            if ( d < dist ) {
                dist = d;
                ref_length = line.length;
                best_index = i;
            }
        }

        if ( dist >= max_dist ) {
            dist = rT(max_dist);
            if ( best_match_index ) *best_match_index = INVALID_BEST_MATCH_INDEX;
            residual[0] = dist;
        } else {
            if (best_match_index) *best_match_index = best_index;

            // Weight.
            double long_length = std::max(tst_line->length, ref_length);
            if (weight_sigma > 0)
                //            residual[0] = dist * rT( std::exp(long_length/weight_sigma) ) ;
                residual[0] = dist * (1 + 10 * (1 - std::exp(-long_length / weight_sigma)));
            else
                residual[0] = dist;
        }

        return true;
    }

private:
    static const int INVALID_BEST_MATCH_INDEX;
    const std::vector<SimpleLine>* ref_lines; // The reference lines.
    const SimpleLine* tst_line;               // A test line.
    double max_dist;                          // Distance threshold for cost clipping.
    double weight_sigma = 100;                // Weight coefficient.
    int* best_match_index;                    // Line index of the best matching reference line.
};

// Create the static const member of CF_SingleLine.
const int CF_SingleLine::INVALID_BEST_MATCH_INDEX = -1;

/**
 * Build the optimization problem.
 * @param ref_lines A collection of lines in the reference image.
 * @param tst_lines A collection of lines in the test image.
 * @param problem The optimization problem.
 * @param best_match_indices The resulting best matching reference line indices.
 * @param hg The 3x3 homography matrix.
 * @param weight_sigma Weight coefficient.
 */
static void build_op_problem(
        const std::vector<SimpleLine>& ref_lines,
        const std::vector<SimpleLine>& tst_lines,
        ceres::Problem& problem,
        std::vector<int>& best_match_indices,
        cv::Mat& hg,
        double weight_sigma) {
    // Clear best_match_indices.
    const int N = tst_lines.size();
    best_match_indices.resize(N);

    // Allocate solution.
    hg = cv::Mat::eye( 3, 3, CV_64FC1 );

    // Build the problem.
    for ( int i = 0; i < N; i++ ) {
        const auto& tst_line = tst_lines[i];
        ceres::CostFunction* cf =
                new ceres::AutoDiffCostFunction<CF_SingleLine, 1, 9>(
                        new CF_SingleLine( &ref_lines, &tst_line, 100, weight_sigma, &best_match_indices[i] ) );
        problem.AddResidualBlock( cf, nullptr, hg.ptr<double>() );
    }
}

/**
 * Draw line correspondences between two images. The horizontal lines will be marked in green. And blue
 * for the vertical lines.
 * @param ref_img Reference/target/training image.
 * @param tst_img Test/source image.
 * @param ref_lines Lines in the reference image.
 * @param tst_lines Lines in the test image.
 * @param tst_2_ref_indices Best matching reference line indices.
 * @return A new image showing the line correspondences.
 */
static cv::Mat draw_line_correspondence(
        const cv::Mat& ref_img,
        const cv::Mat& tst_img,
        const std::vector<SimpleLine>& ref_lines,
        const std::vector<SimpleLine>& tst_lines,
        const std::vector<int>& tst_2_ref_indices) {
    // Assuming that the image sizes of the two images are the same.
    const int rows = ref_img.rows;
    const int cols = ref_img.cols;
    assert( rows == tst_img.rows );
    assert( cols == tst_img.cols );

    // Convert image if necessary.
    const cv::Mat img0 = iu::grey_2_BGR(ref_img);
    const cv::Mat img1 = iu::grey_2_BGR(tst_img);

    // Create the canvas.
    cv::Mat canvas = cv::Mat::zeros(2*rows, 2*cols, CV_8UC3);

    // Copy the original images to the canvas.
    img0.copyTo( canvas( cv::Rect(    0,    0, cols, rows ) ) );
    img1.copyTo( canvas( cv::Rect( cols,    0, cols, rows ) ) );
    img1.copyTo( canvas( cv::Rect(    0, rows, cols, rows ) ) );

    // Draw all the correspondences.
    for ( int i = 0; i < tst_2_ref_indices.size(); i++ ) {
        const int ref_index = tst_2_ref_indices[i];
        if ( ref_index < 0 ) continue; // No matched correspondence.

        // Get the lines.
        const SimpleLine& ref_line = ref_lines[ref_index];
        const SimpleLine& tst_line = tst_lines[i];

        // Middle point.
        const auto [ mp_ref_x, mp_ref_y ] = ref_line.middle();
        const auto [ mp_tst_x, mp_tst_y ] = tst_line.middle();

        // Check the angle.
        double angle = ref_line.angle;
        if ( angle > PI ) angle -= PI;

        // Coordinate shifts for the tst entities.
        int shift_x = 0;
        int shift_y = 0;
        cv::Scalar color;

        if ( angle >= PI/4 && angle < 3*PI/4) {
            // Vertical line, do left-right plot.
            shift_x += cols;
            color = cv::Scalar(209, 49, 31);
        } else {
            // Horizontal line, do top-bottom plot.
            shift_y += rows;
            color = cv::Scalar(18, 219, 44);
        }

        // Draw the line and the end points.
        cv::line( canvas,
                  cv::Point( static_cast<int>(mp_ref_x),           static_cast<int>(mp_ref_y) ),
                  cv::Point( static_cast<int>(mp_tst_x) + shift_x, static_cast<int>(mp_tst_y) + shift_y ),
                  color, 2 );
        cv::circle(canvas,
                   cv::Point( static_cast<int>(mp_ref_x), static_cast<int>(mp_ref_y) ),
                   10,
                   color,
                   cv::FILLED);
        cv::circle(canvas,
                   cv::Point( static_cast<int>(mp_tst_x) + shift_x, static_cast<int>(mp_tst_y) + shift_y ),
                   10,
                   color,
                   cv::FILLED);
    }

    return canvas;
}

/**
 * Test LSD line detector and optimization based line matching.
 * @param mi The model input.
 */
static void test_lsd_optimization(
        const MatchInput& mi) {

    QUICK_TIME_START(match)
    // Create the masks.
    cv::Mat mask0 = cv::Mat::ones( mi.img0.size(), CV_8UC1 );
    cv::Mat mask1 = cv::Mat::ones( mi.img0.size(), CV_8UC1 );

    // LSD detector.
    cv::Ptr<cvline::LSDDetector> lsd = cvline::LSDDetector::createLSDDetector();

    // Detect lines.
    std::vector<cvline::KeyLine> keylines0, keylines1;
    lsd->detect( mi.img0, keylines0, 2, 2, mask0 );
    lsd->detect( mi.img1, keylines1, 2, 2, mask1 );

//    // Show the info of these key lines.
//    std::cout << "keyliens0: \n";
//    show_keyline_info( keylines0 );
//    std::cout << "\nkeyliens1: \n";
//    show_keyline_info( keylines1 );

    std::cout << "keylines0.size() = " << keylines0.size() << "\n";
    std::cout << "keylines1.size() = " << keylines1.size() << "\n";

    // Filter the lines.
    std::vector<cvline::KeyLine> filtered_lines_0, filtered_lines_1;
    filter_keylines( keylines0, filtered_lines_0, 0, 2, mi.keyline_filter );
    filter_keylines( keylines1, filtered_lines_1, 0, 2, mi.keyline_filter );

    // Convert the KeyLine objects to SimpleLine objects.
    std::vector<SimpleLine> simple_lines_0 = collect_and_convert_keylines( filtered_lines_0 );
    std::vector<SimpleLine> simple_lines_1 = collect_and_convert_keylines( filtered_lines_1 );

    // Ceres problem.
    cv::Mat hg;
    ceres::Problem problem;
    std::vector<int> best_match_indices;
    build_op_problem(
            simple_lines_0, simple_lines_1,
            problem, best_match_indices,
            hg, mi.weight_sigma );

    // Configure the solver.
    ceres::Solver::Options options;
    options.max_num_iterations = 10000;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 4;

    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    options.function_tolerance  = 1e-6;
    options.gradient_tolerance  = 1e-10;
    options.parameter_tolerance = 1e-8;

    // Solve.
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    QUICK_TIME_SHOW(match, "match in")

    // Show solution details.
    std::cout << summary.FullReport() << std::endl;

    // Show the raw homography matrix.
    std::cout << "hg = \n" << hg << "\n";

    // Normalize the homography matrix.
    hg /= hg.at<double>(2, 2);
    std::cout << "hg normalized = \n" << hg << "\n";

    // Convert the gray image into BGR.
    cv::Mat img0_bgr = iu::grey_2_BGR(mi.img0);
    cv::Mat img1_bgr = iu::grey_2_BGR(mi.img1);

    // Apply homography.
    cv::Mat warped;
    cv::warpPerspective( img1_bgr, warped, hg, cv::Size( img0_bgr.cols, img0_bgr.rows ), cv::INTER_CUBIC);

    // Merge images.
    cv::Mat dummy = warped.clone();
    dummy.setTo(cv::Scalar(0, 1, 0));
    cv::Mat green;
    cv::multiply(warped, dummy, green);
    cv::Mat merged;
    cv::scaleAdd(img0_bgr, 0.5, green, merged);

    // Write.
    std::stringstream ss;
    ss << "./" << mi.name << "_merged_optimization.png";
    cv::imwrite(ss.str(), merged);

    // ========== Debug use. ==========
    std::cout << "filtered_lines_0.size() = " << filtered_lines_0.size() << "\n";
    std::cout << "filtered_lines_1.size() = " << filtered_lines_1.size() << "\n";

    // Draw the key lines for debugging.
    draw_keyline_mi(mi, filtered_lines_0, filtered_lines_1);

    // Print the best matches.
    std::cout << "Best match indices: \n";
    for ( int i = 0; i < best_match_indices.size(); i++ ) {
        std::cout << i << " -> " << best_match_indices[i] << "\n";
    }

    // Draw line correspondence.
    cv::Mat correspondence_image = draw_line_correspondence(
            mi.img0, mi.img1,
            simple_lines_0, simple_lines_1,
            best_match_indices);

    // Save the correspondence image.
    ss.str(""); ss.clear();
    ss << "./" << mi.name << "_NearestLineCorrespondence.png";
    cv::imwrite(ss.str(), correspondence_image);
}

int main(int argc, char** argv) {
    std::cout << "Hello, test_lsd_matching! \n";

    cv::CommandLineParser parser(argc, argv,
                          "{help h usage ?     |         | print this message}"
                          "{@template_img      |         | the template image}"
                          "{@test_img          |         | the test image}"
                          "{name               |  match  | the name of the case}"
                          "{resize             |    -1   | the longer edge after resizing, use negative value to disable}"
                          "{filter_length      |    10   | the pixel length for filtering the key lines}"
                          "{weight_sigma       |    -1   | the weight sigma value, use negative value to disable}"
                          "{binarise_threshold |    -1   | the threshold for binarization, use negative value to disable}"
                          "{contrast           |    -1   | the contrast coefficient, use negative value to disable}"
                          "{brightness         |     0   | the added brightness, use zero to disable}");
    parser.about("Line matching by optimization.");

    const auto fn_img_0 = parser.get<std::string>("@template_img");
    const auto fn_img_1 = parser.get<std::string>("@test_img");
    const auto name     = parser.get<std::string>("name");
    const auto resize_longer_edge = parser.get<int>("resize");
    const auto filter_length      = parser.get<int>("filter_length");
    const auto weight_sigma       = parser.get<double>("weight_sigma");
    const auto binarise_threshold = parser.get<int>("binarise_threshold");
    const auto contrast           = parser.get<double>("contrast");
    const auto brightness         = parser.get<double>("brightness");

    // Show the input arguments.
    std::cout << "fn_img_0: " << fn_img_0 << "\n";
    std::cout << "fn_img_1: " << fn_img_1 << "\n";
    std::cout << "name: " << name << "\n";
    std::cout << "resize_longer_edge: " << resize_longer_edge << "\n";
    std::cout << "filter_length: " << filter_length << "\n";
    std::cout << "weight_sigma: " << weight_sigma << "\n";
    std::cout << "binarise_threshold: " << binarise_threshold << "\n";

    MatchInput mi( name, resize_longer_edge, filter_length, weight_sigma );
    mi.read_img0(fn_img_0);
    mi.read_img1(fn_img_1);

    if ( contrast > 0 ) {
        mi.img0 = iu::adjust_contrast_brightness(mi.img0, contrast, brightness);
        mi.img1 = iu::adjust_contrast_brightness(mi.img1, contrast, brightness);
    }

    if (binarise_threshold > 0) {
        // Binaries.
        mi.img0 = iu::binaries(mi.img0, binarise_threshold);
        mi.img1 = iu::binaries(mi.img1, binarise_threshold);

        // Dilate-erode.
        mi.img0 = iu::dilate_erode(mi.img0);
        mi.img1 = iu::dilate_erode(mi.img1);

        // Additional blur.
        cv::GaussianBlur(mi.img0, mi.img0, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
        cv::GaussianBlur(mi.img1, mi.img1, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    }

    // LSD and optimization.
    test_lsd_optimization(mi);

    return 0;
}
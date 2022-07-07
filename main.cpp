//
// Created by qzj on 2020/8/4.
//
#include<opencv2/core/core.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "ExtractLineSegment.h"

using namespace cv;

int main(int argc, char **argv)
{

    cv::Mat image = cv::imread("./data/1.jpeg", CV_LOAD_IMAGE_UNCHANGED);

    StructureSLAM::LineSegment* pLineSegment = new StructureSLAM::LineSegment();

    Mat mLdesc;
    vector<KeyLine> mvKeylinesUn;
    vector<Vector3d> mvKeyLineFunctions;    //特征线段所在直线的系数
    pLineSegment->ExtractLineSegment(image, mvKeylinesUn, mLdesc, mvKeyLineFunctions);

    //绿色
#ifdef LSD_REFINE
    drawKeylines(image, mvKeylinesUn, image, 2, Scalar(0, 0, 255));     //自己添加的，绘制特征线
#else
    drawKeylines(image, mvKeylinesUn, image, Scalar(0, 0, 255));     //自己添加的，绘制特征线
#endif
    cv::resize(image, image,cv::Size(image.cols/4, image.rows/4));
    cv::imshow("line",image);
    waitKey(0);
    //cv::imwrite("re.png",image);

    return 0;
}
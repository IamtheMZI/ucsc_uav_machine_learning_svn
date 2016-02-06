#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

#define TRAIN_NUM 4
#define TRAIN_SIZE 2

int main(int, char**)
{
    // Data for visual representation
	Scalar Black = Scalar(0,0,0),White = Scalar(255,255,255),Grey = Scalar(128,128,128);
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    int labels[4] = {1, 2, -1, -1};
    Mat labelsMat(4, 1, CV_32SC1, labels);

    float trainingData[TRAIN_NUM][TRAIN_SIZE] = { {50, 10}, {25, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    SVM::Params params;
    params.svmType    = SVM::C_SVC;
    params.kernelType = SVM::LINEAR;
    params.termCrit   = TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);

    // Train the SVM
    Ptr<SVM> svm = StatModel::train<SVM>(trainingDataMat, ROW_SAMPLE, labelsMat, params);

    Vec3b green(0,255,0), blue (255,0,0), red(0,0,255);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = svm->predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
			else if (response == 2)
                image.at<Vec3b>(i,j)  = red;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
	for(int i=0; i< TRAIN_NUM; i++){
		if(labels[i] == 1)
		circle( image, Point(trainingData[i][0],  trainingData[i][1]), 5, White, thickness, lineType );
		else if(labels[i] == -1)
		circle( image, Point(trainingData[i][0],  trainingData[i][1]), 5, Black, thickness, lineType );
	else if(labels[i] == 2)
		circle( image, Point(trainingData[i][0],  trainingData[i][1]), 5, Grey, thickness, lineType );
	}

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getSupportVectors();

    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image

    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);

}
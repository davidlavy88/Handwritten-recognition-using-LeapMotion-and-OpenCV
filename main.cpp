//Example using the file digits.png from opencv/samples/python2/data
// it has 5000 handwritten digits (500 for each digit). Each digit is a 20x20 image
//We'll be using the simplest feature which is just flatten the image into a single
//row with 400 pixels

#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <ctype.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include "ml.h"

using namespace std;

const cv::Size smallsize(20,20);

void PreProcessImage(cv::Mat *inImage, cv::Mat *outImage); //TODO: for better recognition using actual features
void LearnFromImages(cv::Mat& input, cv::Mat& trainData, cv::Mat& trainClasses, cv::Mat& testData, cv::Mat& testClasses, vector<cv::Mat> &vecTrain,  vector<cv::Mat> &vecTest);
void VectorOfImages(cv::Mat& input, vector<cv::Mat>& trainData, vector<cv::Mat>& testData);
void RunSelfTest(cv::Mat samples_train, cv::KNearest& knn2);
void AnalyseImage(cv::Mat test_data, cv::KNearest& knearest);

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(int argc, char** argv)
{
    const int K = 11;
    cv::Mat img, outfile, dig;
    cv::Mat test_digits(200,20,CV_32FC1);
    vector<cv::Mat> vecTrData, vecTsData;
    img = cv::imread(argv[1],1);
    PreProcessImage(&img, &outfile);
    for (int a=0; a<10; a++)
    {
        cv::Rect rect = cv::Rect(0,a*100,smallsize.width,smallsize.height);
        dig = cv::Mat(outfile,rect);
        dig.copyTo(test_digits(cv::Rect(0,a*20,smallsize.width,smallsize.height)));
    }
    cv::imshow("Samples",test_digits);
    cv::Mat trainData = cv::Mat(2500,400, CV_32FC1);
    cv::Mat trainClasses = cv::Mat(2500, 1, CV_32FC1);
    cv::Mat testData = cv::Mat(2500,400, CV_32FC1);
    cv::Mat testClasses = cv::Mat(2500, 1, CV_32FC1);
    LearnFromImages(outfile,trainData,trainClasses,testData,testClasses,vecTrData,vecTsData);
    CvMat ttrainData = trainData;
    CvMat ttrainClasses = trainClasses;
    CvMat ttestData = testData;
    CvMat ttestClasses = testClasses;
    cv::KNearest knearest(&ttrainData,&ttrainClasses);
    //Testing the KNearest object
    cv::Mat digit2;
    CvMat* sample = cvCreateMat(1,400,CV_32FC1);
    for (int z=0;z<10;z++)
    {
        cv::Rect rect = cv::Rect(0,z*20,20,20);
        digit2 = cv::Mat(test_digits,rect);
        *sample = digit2.clone().reshape(0,1);
        float detectedClass = knearest.find_nearest(sample,5);
        if (z != (int) ((detectedClass)))
        {
            cout << "False. It's " << z << "but it gave me " << (int) ((detectedClass));
            exit(1);
        }
        cout << "Correct! " << (int) ((detectedClass)) << endl;
    }
//    RunSelfTest(test_digits,knearest);
//    AnalyseImage(testData,testClasses,knearest);
    //Using it in real samples
    float right(0), wrong(0);
    for (int p=0; p<testData.rows; p++)
    {
        *sample = testData(cv::Rect(0,p,400,1));
        float detectedClass = knearest.find_nearest(sample,5);
        if ((int) ((testClasses.at<float>(0,p))) != (int) ((detectedClass)))
        {
            //cout << "False. It's " << (int) ((testClasses.at<float>(0,p))) << " but it gave me " << (int) ((detectedClass)) << endl;
            wrong++;
            //exit(1);
        }
        //cout << "Correct! " << (int) ((detectedClass)) << endl;
        else right++;
        //cv::imshow("Correct number",vecTsData[p]);  // USE THIS IF YOU WANT TO TEST IT ON
        //cv::waitKey(0);                             // JUST A SUBSET OF SAMPLES
    }
    cout << "Right: " << right << endl;
    cout << "Wrong: " << wrong << endl;
    cout << "The accuracy of this algorithm is: " << (float) (right*100.0/testData.rows) << "% " << endl;
    /*int rnd_image = atoi(argv[2]);
    cv::imshow("Image sample",digits_shuf[rnd_image]);
    //cout << labels_shuf[rnd_image] << endl << endl;
    cout << "N. detected: " << number_detected << " " << labels_shuf[2500] << endl << endl;
    cout << "rows: " << (int)img_labels_shuf.at<uchar>(rnd_image,0) << endl << endl;
    cv::FileStorage file("../results.xml", cv::FileStorage::WRITE);
    // Write to file!
    file << "Result" << result;
    file.release();*/
    cv::waitKey(0);
    return 0;
}

void PreProcessImage(cv::Mat *inImage,cv::Mat *outImage)
{
    cv::Mat image, image_float;
    cv::cvtColor(*inImage, image, CV_BGR2GRAY);
    image.convertTo(image_float, CV_32FC1);
    *outImage = image_float.clone();
}

void LearnFromImages(cv::Mat& input, cv::Mat& trainData, cv::Mat& trainClasses, cv::Mat& testData, cv::Mat& testClasses, vector<cv::Mat> &vecTrain,  vector<cv::Mat> &vecTest)
{
    //Split image into 5000 cells and assign 50/50 to trainData and testData
    cv::Mat dig_vector, featmat(5000,400,input.type()); //Image of flatten digits stacked. Size = (2500,400)
    vector<cv::Mat> digits; //Vector of images
    //Loop over the image to extract them one by one
    for (int i=0; i<input.rows; i+=smallsize.height)
    {
        for (int j=0; j<input.cols; j+=smallsize.width)
        {
            cv::Rect rect = cv::Rect(j,i,smallsize.width,smallsize.height); //Extract digit
            digits.push_back(cv::Mat(input,rect)); //Store digits as images
            dig_vector = cv::Mat(input,rect).clone().reshape(0,1); //Flat the digit to store it
            dig_vector.copyTo(featmat(cv::Rect(0,(input.cols/smallsize.width)*i/smallsize.height+j/smallsize.height,dig_vector.cols,dig_vector.rows)));
        }
    }
    vector<int> labels;
    cv::Mat img_labels = cv::Mat::eye(5000,1,input.type());
    for (int i=0;i<10;i++)
    {
        for (int j=0;j<500;j++)
        {
            labels.push_back(i);
            img_labels.at<int>(i*500+j,0) = i;
        }
    }
    //Now we will shuffle the feature matrix, vectors of digits and labels
    vector<cv::Mat> digits_shuf;
    vector<int> indexes, labels_shuf;
    cv::Mat feat_shuf, img_labels_shuf;
    indexes.reserve(labels.size());
    for (int i=0;i<labels.size();++i)
        indexes.push_back(i);
    random_shuffle(indexes.begin(), indexes.end());
    for (vector<int>::iterator it1 = indexes.begin(); it1!=indexes.end();++it1)
    {
        feat_shuf.push_back(featmat.row(*it1));
        img_labels_shuf.push_back(img_labels.at<uchar>(*it1,0));
        digits_shuf.push_back(digits[*it1]);
        labels_shuf.push_back(labels[*it1]);
    }
    //Assign 50-50 samples to train and test
    trainData = feat_shuf(cv::Rect(0,0,feat_shuf.cols,feat_shuf.rows/2));
    testData = feat_shuf(cv::Rect(0,feat_shuf.rows/2,feat_shuf.cols,feat_shuf.rows/2));
    trainClasses = img_labels_shuf(cv::Rect(0,0,img_labels_shuf.cols,img_labels_shuf.rows/2));
    testClasses = img_labels_shuf(cv::Rect(0,img_labels_shuf.rows/2,img_labels_shuf.cols,img_labels_shuf.rows/2));
    trainClasses.convertTo(trainClasses, CV_32FC1);
    testClasses.convertTo(testClasses, CV_32FC1);
    vector<cv::Mat>::const_iterator first_trainD = digits_shuf.begin();
    vector<cv::Mat>::const_iterator end_trainD = digits_shuf.begin()+2500;
    vector<cv::Mat>::const_iterator first_testD = digits_shuf.begin()+2501;
    vector<cv::Mat>::const_iterator end_testD = digits_shuf.end();
    vector<cv::Mat> ata(first_trainD,end_trainD);
    vector<cv::Mat> eta(first_testD,end_testD);
    vecTrain.reserve(ata.size());
    vecTest.reserve(eta.size());
    copy(ata.begin(),ata.end(),back_inserter(vecTrain));
    copy(eta.begin(),eta.end(),back_inserter(vecTest));
}

void VectorOfImages(cv::Mat& input, vector<cv::Mat> &trainData,  vector<cv::Mat> &testData)
{
    vector<cv::Mat> digits; //Vector of images
    for (int i=0; i<input.rows; i+=smallsize.height)
    {
        for (int j=0; j<input.cols; j+=smallsize.width)
        {
            cv::Rect rect = cv::Rect(j,i,smallsize.width,smallsize.height); //Extract digit
            digits.push_back(cv::Mat(input,rect)); //Store digits as images
        }
    }
    vector<int> labels;
    for (int i=0;i<10;i++)
    {
        for (int j=0;j<500;j++)
        {
            labels.push_back(i);
        }
    }
    vector<cv::Mat> digits_shuf;
    vector<int> indexes, labels_shuf;
    indexes.reserve(labels.size());
    for (int i=0;i<labels.size();++i)
        indexes.push_back(i);
    random_shuffle(indexes.begin(), indexes.end());
    for (vector<int>::iterator it1 = indexes.begin(); it1!=indexes.end();++it1)
    {
        digits_shuf.push_back(digits[*it1]);
        labels_shuf.push_back(labels[*it1]);
    }
    //Assign 50-50 samples to train and test
    vector<cv::Mat>::const_iterator first_trainD = digits_shuf.begin();
    vector<cv::Mat>::const_iterator end_trainD = digits_shuf.begin()+2500;
    vector<cv::Mat>::const_iterator first_testD = digits_shuf.begin()+2501;
    vector<cv::Mat>::const_iterator end_testD = digits_shuf.end();
    vector<cv::Mat> ata(first_trainD,end_trainD);
    vector<cv::Mat> eta(first_testD,end_testD);
    trainData.reserve(ata.size());
    testData.reserve(eta.size());
    copy(ata.begin(),ata.end(),back_inserter(trainData));
    copy(eta.begin(),eta.end(),back_inserter(testData));
}

void RunSelfTest(cv::Mat& samples_train, cv::KNearest& knn2)
{
    cv::Mat img, digit;
    CvMat* sample = cvCreateMat(1,400,CV_32FC1);
    for (int z=0;z<10;z++)
    {
        cv::Rect rect = cv::Rect(0,z*20,20,20);
        digit = cv::Mat(samples_train,rect);
        *sample = digit.clone().reshape(0,1);
        float detectedClass = knn2.find_nearest(sample,5);
        if (z != (int) ((detectedClass)))
        {
            cout << "False. It's " << z << "but it gave me " << (int) ((detectedClass));
            exit(1);
        }
        cout << "Correct! " << (int) ((detectedClass)) << endl;
        cv::imshow("Digit", digit);
    }
}

void AnalyseImage(cv::Mat test_data, cv::Mat test_classes, cv::KNearest& knearest)
{
    CvMat* sample2 = cvCreateMat(1, 400, CV_32FC1);
    for (int p=0; p<20; p++)
    {
        *sample2 = test_data(cv::Rect(0,p,400,1));
        float detectedClass = knearest.find_nearest(sample2,5);
        if ((int) ((test_classes.at<float>(0,p))) != (int) ((detectedClass)))
        {
            cout << "False. It's " << (int) ((test_classes.at<float>(0,p))) << "but it gave me " << (int) ((detectedClass));
            exit(1);
        }
        cout << "Correct! " << (int) ((detectedClass)) << endl;
    }
}

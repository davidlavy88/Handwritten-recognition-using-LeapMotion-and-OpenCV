//Example using the file digits.png from opencv/samples/python2/data
// it has 5000 handwritten digits (500 for each digit). Each digit is a 20x20 image
//We'll be using the simplest feature which is just flatten the image into a single
//row with 400 pixels

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <ctype.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <iomanip>
#include <math.h>
#include <complex>
#include <numeric>
#include <cfloat>
#include <iterator>
#include <dirent.h>
#include "Leap.h"

using namespace std;
using namespace Leap;

void NormResample(vector<vector<float> > &iniQueryVector, vector<vector<vector<float>>> &oldLetters, vector<vector<vector<float>>> &newLetters);
void ExtractXYcomp(vector<vector<float>> &character, vector<float> &compx, vector<float> &compy);
void InsertXYcomp(vector<float> &compx, vector<float> &compy, vector<vector<float>> &new_character);
vector< float > interp1( vector< float > &x, vector< float > &y, vector< float > &x_new );
void ExtractDFTCoeff(vector<vector<vector<float>>> &new_characters, cv::Mat &trainData);
//vector<vector<float>> ExtractDFTCoeff(vector<vector<float>> &new_character);
int findNearestNeighbourIndex( float value, vector< float > &x );

ofstream output_file;
ifstream input_file;
float finger_array_x[2000];
float finger_array_y[2000];
float finger_array_z[2000];
vector<vector<float>> finger_2d(2000, vector<float>(2, 0));

int array_counter = 0;
int finger_id = 0;

class SampleListener : public Listener {
public:
    virtual void onInit(const Controller&);
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onExit(const Controller&);
    virtual void onFrame(const Controller&);
    virtual void onFocusGained(const Controller&);
    virtual void onFocusLost(const Controller&);
    virtual void onDeviceChange(const Controller&);
    virtual void onServiceConnect(const Controller&);
    virtual void onServiceDisconnect(const Controller&);

private:
};

const std::string fingerNames[] = { "Thumb", "Index", "Middle", "Ring", "Pinky" };
const std::string boneNames[] = { "Metacarpal", "Proximal", "Middle", "Distal" };
const std::string stateNames[] = { "STATE_INVALID", "STATE_START", "STATE_UPDATE", "STATE_END" };

void SampleListener::onInit(const Controller& controller) {
    std::cout << "Initialized" << std::endl;
}

void SampleListener::onConnect(const Controller& controller) {
    std::cout << "Connected" << std::endl;
    controller.enableGesture(Gesture::TYPE_CIRCLE);
    controller.enableGesture(Gesture::TYPE_KEY_TAP);
    controller.enableGesture(Gesture::TYPE_SCREEN_TAP);
    controller.enableGesture(Gesture::TYPE_SWIPE);
}

void SampleListener::onDisconnect(const Controller& controller) {
    // Note: not dispatched when running in a debugger.
    std::cout << "Disconnected" << std::endl;
}

void SampleListener::onExit(const Controller& controller) {
    std::cout << "Exited" << std::endl;
}

void SampleListener::onFrame(const Controller& controller) {

    // Get the most recent frame and report some basic information
    const Frame frame = controller.frame();
    HandList hands = frame.hands();
    for (HandList::const_iterator hl = hands.begin(); hl != hands.end(); ++hl) {
        // Get the first hand
        const Hand hand = *hl;
        std::string handType = hand.isLeft() ? "Left hand" : "Right hand";
        //std::cout << std::string(2, ' ') << handType << ", id: " << hand.id()
        //          << ", palm position: " << hand.palmPosition() << std::endl;
        // Get the hand's normal vector and direction
        const Vector normal = hand.palmNormal();
        const Vector direction = hand.direction();

        // Get fingers
        const FingerList fingers = hand.fingers();
        const Finger finger = *(++fingers.begin()); // get Index finger
        std::cout << std::string(4, ' ') << fingerNames[finger.type()]
            << " finger, id: " << finger.id()
            << ", length: " << finger.length()
            << "mm, width: " << finger.width() << std::endl;

        Bone::Type boneType = static_cast<Bone::Type>(3); // Get distal bone for character recognization
        Bone bone = finger.bone(boneType);
        //output_file << finger.id()
        //<< ", " << bone.nextJoint().x << ", " << bone.nextJoint().y << ", " << bone.nextJoint().z << std::endl;

        // 2D-3D transformation
        if (finger.id() != finger_id) {
            finger_id = finger.id(); // reset the data
            array_counter = 0;
        }
        else array_counter++;
        finger_array_x[array_counter] = bone.nextJoint().x;
        finger_array_y[array_counter] = bone.nextJoint().y;
        finger_array_z[array_counter] = bone.nextJoint().z;
    }
}

void SampleListener::onFocusGained(const Controller& controller) {
    std::cout << "Focus Gained" << std::endl;
}

void SampleListener::onFocusLost(const Controller& controller) {
    std::cout << "Focus Lost" << std::endl;
}

void SampleListener::onDeviceChange(const Controller& controller) {
    std::cout << "Device Changed" << std::endl;
    const DeviceList devices = controller.devices();

    for (int i = 0; i < devices.count(); ++i) {
        std::cout << "id: " << devices[i].toString() << std::endl;
        std::cout << "  isStreaming: " << (devices[i].isStreaming() ? "true" : "false") << std::endl;
    }
}

void SampleListener::onServiceConnect(const Controller& controller) {
    std::cout << "Service Connected" << std::endl;
}

void SampleListener::onServiceDisconnect(const Controller& controller) {
    std::cout << "Service Disconnected" << std::endl;
}

void transform_2d() {
    // DO THIS
    int i;
    int id = 0;
    int next_i = 0;

    float sum_x = 0;
    float sum_y = 0;
    float sum_z = 0;
    float sum_xx = 0;
    float sum_xy = 0;
    float sum_yy = 0;
    float sum_xz = 0;
    float sum_yz = 0;
    float sum = 0;

    float rand_vec[3] = { 1, 0, 1 }; //random vector

    for (i = 0; i <= array_counter; i++) {
        if (i == 0) {
            sum_x = finger_array_x[i];
            sum_y = finger_array_y[i];
            sum_z = finger_array_z[i];
            sum_xx = finger_array_x[i] * finger_array_x[i];
            sum_xy = finger_array_x[i] * finger_array_y[i];
            sum_xz = finger_array_x[i] * finger_array_z[i];
            sum_yy = finger_array_z[i] * finger_array_z[i];
            sum_yz = finger_array_y[i] * finger_array_z[i];
            sum = 1;
        }
        else {
            sum_x = sum_x + finger_array_x[i];
            sum_y = sum_y + finger_array_y[i];
            sum_z = sum_z + finger_array_z[i];
            sum_xx = sum_xx + finger_array_x[i] * finger_array_x[i];
            sum_xy = sum_xy + finger_array_x[i] * finger_array_y[i];
            sum_xz = sum_xz + finger_array_x[i] * finger_array_z[i];
            sum_yy = sum_yy + finger_array_y[i] * finger_array_y[i];
            sum_yz = sum_yz + finger_array_y[i] * finger_array_z[i];
            sum = sum + 1;
        }
    }

    float det = sum_xx * (sum_yy * sum - sum_y * sum_y) - sum_xy * (sum_xy * sum - sum_x * sum_y) + sum_x * (sum_xy * sum_y - sum_x * sum_yy);

    float x_11 = (sum_yy * sum - sum_y * sum_y) / det;
    float x_12 = (sum_x * sum_y - sum_xy * sum) / det;
    float x_13 = (sum_xy * sum_y - sum_x * sum_yy) / det;
    float x_21 = (sum_x * sum_y - sum_xy * sum) / det;
    float x_22 = (sum_xx * sum - sum_x * sum_x) / det;
    float x_23 = (sum_x * sum_xy - sum_xx * sum_y) / det;
    float x_31 = (sum_xy * sum_y - sum_yy * sum_x) / det;
    float x_32 = (sum_xy * sum_x - sum_xx * sum_y) / det;
    float x_33 = (sum_xx * sum_yy - sum_xy * sum_xy) / det;
    //Z = X1 * Y;
    float z_1 = x_11 * sum_xz + x_12 * sum_yz + x_13 * sum_z;
    float z_2 = x_21 * sum_xz + x_22 * sum_yz + x_23 * sum_z;
    float z_3 = x_31 * sum_xz + x_32 * sum_yz + x_33 * sum_z;
    float z_norm = sqrt(z_1 * z_1 + z_2 * z_2 + z_3 * z_3);
    z_1 = z_1 / z_norm;
    z_2 = z_2 / z_norm;
    z_3 = z_3 / z_norm;
    float a = z_1 + z_3; //dot product with rand vector 1 0 1
    float b_1 = 1 - a*z_1;
    float b_2 = 0 - a*z_2;
    float b_3 = 1 - a*z_3;
    float b_norm = sqrt(b_1 * b_1 + b_2 * b_2 + b_3 * b_3);
    b_1 = b_1 / b_norm;
    b_2 = b_2 / b_norm;
    b_3 = b_3 / b_norm;
    //c = cross(b, Z);
    float c_1 = b_2 * z_3 - b_3 * z_2;
    float c_2 = b_3 * z_1 - b_1 * z_3;
    float c_3 = b_1 * z_2 - b_2 * z_1;
    float c_norm = sqrt(c_1 * c_1 + c_2 * c_2 + c_3 * c_3);
    c_1 = c_1 / c_norm;
    c_2 = c_2 / c_norm;
    c_3 = c_3 / c_norm;
    //b = b / sqrt(b(1) ^ 2 + b(2) ^ 2 + b(3) ^ 2);
    //c = c / sqrt(c(1) ^ 2 + c(2) ^ 2 + c(3) ^ 2); % either positive or negative
    int y_neg_cnt = 0;
    finger_2d = vector<vector<float>>(2000, vector<float>(2, 0)); // reset data
    for (i = 0; i <= array_counter; i++) {
        // NO ROUNDING PART //////////////////////////
        //finger_2d[i][0] = b_1 * finger_array_x[i] + b_2 * finger_array_y[i] + b_3 * finger_array_z[i];
        //finger_2d[i][1] = c_1 * finger_array_x[i] + c_2 * finger_array_y[i] + c_3 * finger_array_z[i];
        //////////////////////////////////////////////
        // ROUNDING PART /////////////////////////////
        finger_2d[i][0] = roundf((b_1 * finger_array_x[i] + b_2 * finger_array_y[i] + b_3 * finger_array_z[i]) * 100) / 100;
        finger_2d[i][1] = -roundf((c_1 * finger_array_x[i] + c_2 * finger_array_y[i] + c_3 * finger_array_z[i]) * 100) / 100;
        if (finger_2d[i][1] < 0) y_neg_cnt = y_neg_cnt + 1;
    }

    if (y_neg_cnt == (array_counter + 1)) {//if all y point is negative, flip it
        for (i = 0; i <= array_counter; i++) {
            finger_2d[i][1] = -finger_2d[i][1];
        }
    }
    /////////////////////////////////////////////////////////////////////////
    // Writing testing data to output files, UNCOMMENT IF DON'T WANT ANYMORE
    for (i = 0; i <= array_counter; i++) {
        output_file << finger_id
            << " " << finger_2d[i][0] << " " << finger_2d[i][1] << std::endl;
    }
    ////////////////////////////////////////////////////////////////////////
}


int main()
{
    // Create a sample listener and controller
    SampleListener listener;
    Controller controller;


    string input;
    vector<string> all;
    vector<vector<vector<float>>> oldLetters, newLetters;
    vector<vector<float>> data, iniQueryVector;
    vector<float> query;
    ifstream infile;
    int finish(0), idxTrain(0);
    cout << "////////////////////////////////////" << endl;
    cout << "      Start training the KNN        " << endl;
    cout << "////////////////////////////////////" << endl;
    float n;
    cout << "Enter number of files to read: ";
    cin >> input;
    int nof = std::stoi(input,nullptr,10);
    //Define the trainClasses matrix
    cv::Mat trainClasses = cv::Mat::eye(nof,1,CV_32F);

    DIR *dir;
    struct dirent *ent;
    //MODIFY WITH CURRENT FOLDER
    if ((dir = opendir ("/home/david/dummy_ws/Project_RT_Final/NN/")) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir (dir)) != NULL) {
        string num(ent->d_name);
//        if ((num[0] != ".") || (num != ".."))
        if (num != ".." ) { if (num != ".")
        {
            n = (float)(num[0]-'0');
//            cout << "You're this digit: " << n << endl;
            //First update the trainClasses matrix
            trainClasses.at<float>(idxTrain,0) = n;
            idxTrain++;

            data.clear(); query.clear();
            float val = 0;
            int size = num.size();
//            printf ("%s %d\n", ent->d_name,size);
            infile.open("/home/david/dummy_ws/Project_RT_Final/NN/"+num);
            all.push_back(num);
            string   line;

            while(getline(infile, line))
            {
                vector<float>   lineData;
                stringstream  linestream(line);

                float value;
                int count(1);
                // Read an integer at a time from the line
                while(linestream >> value)
                {
                    if (count != 1)
                    {
                        // Add the integers from a line to a 1D array (vector)
                        lineData.push_back(value);
                    }
                    count++;
                }
                // When all the integers have been read add the 1D array
                // into a 2D array (as one line in the 2D array)
                data.push_back(lineData);
                query.push_back(val);
                val++;
            }
//            cout << data.size() << endl;
            iniQueryVector.push_back(query);
            oldLetters.push_back(data);
            infile.close();
        }
        }
        else continue;
      }
      closedir (dir);
    } else {
      /* could not open directory */
      perror ("");
      return EXIT_FAILURE;
    }

    cv::Mat trainData(oldLetters.size(),478,CV_32F); //THISSSSS
    NormResample(iniQueryVector, oldLetters,newLetters); //STILL REVIEW FOR DFT
    ExtractDFTCoeff(newLetters,trainData);
    CvMat ttrainData = trainData;
    CvMat ttrainClasses = trainClasses;
    cv::KNearest knearest(&ttrainData,&ttrainClasses);

    cout << "////////////////////////////////////" << endl;
    cout << "         Training complete          " << endl;
    cout << "////////////////////////////////////" << endl;

    cout << "////////////////////////////////////" << endl;
    cout << "         Start testing KNN          " << endl;
    cout << "////////////////////////////////////" << endl;

    char c = 's';
    char i = 'a';
    int user_id = 1;
    int char_count[37];
    for (int j = 0; j < 37; j++) char_count[j] = 1;

    std::string file_name = "id_1_a_1.txt";
    char_count[0] = 2;
    int state = 0; // 0 is stop, 1 is start
    output_file.open(file_name);
    while (c == 'n' || c == 's' || c == 'i') {
        std::cout << "=======================================================================" << std::endl
            << "Current ID: " << user_id << " Current Character: " << i << " Current Trial: ";
        if (i >= 'a' && i <= 'z') {
            std::cout << char_count[i - 97] - 1 << std::endl;
        }
        else {
            std::cout << char_count[i - 22] - 1 << std::endl;
        }
        std::cout << "Press 's' to Start recording" << std::endl
            << "Press 'n' to set a new character" << std::endl
            << "Press 'i' to move to new user" << std::endl
            << "Press other key to quit..." << std::endl;
        std::cin >> c;

        vector<vector<float>> number;
        number.clear();
        data.clear(); query.clear();
        iniQueryVector.clear();
        finger_2d.clear();

        if (c == 'n') {
            std::cout << "Press the character you want to enter: " << std::endl;
            std::cin >> i;
            output_file.close();
            file_name = "id_";
            file_name += user_id + 48;
            file_name += '_';
            file_name += i;
            file_name += '_';
            if (i >= 'a' && i <= 'z') {
                file_name += char_count[i - 97] + 48;
                char_count[i - 97]++;
            }
            else {
                file_name += char_count[i - 22] + 48;
                char_count[i - 22]++;
            }
            file_name += ".txt";

            output_file.open(file_name);
        }
        if (c == 'i') {
            user_id++;
            for (int j = 0; j < 37; j++) char_count[j] = 1;
            char_count[0] = 2;
            i = 'a';
            output_file.close();
            file_name = "id_";
            file_name += user_id + 48;
            file_name += '_';
            file_name += i;
            file_name += '_';
            file_name += char_count[i - 97] + 47;
            file_name += ".txt";
            output_file.open(file_name);
        }
        while (c == 's') {

        if (state == 0) {
              controller.addListener(listener);
              std::cout << "Start Recording..." << std::endl;
              state = 1;
        }
        //std::cin.get();
        else {
            controller.removeListener(listener);
            std::cout << "Stop Recording..." << std::endl;
            cout << "Analizing Data ..." << endl;
            cout << "Before Extracting: " << number.size() << endl;
            transform_2d();

            cout << "Finding Nearest Neighbor ..." << endl;
            //HERE COMES THE PROCESSING
            //Extract the useful data

            float val = 0;
            vector<vector<vector<float>>> testLetters, Letters;
            for (int i=0;i<=array_counter;i++)
            {
                number.push_back(finger_2d[i]);
                query.push_back(val);
                val++;
            }
            cout << "After Extracting: " << number.size() << endl;
            iniQueryVector.push_back(query);
            testLetters.push_back(number);
            cv::Mat testData(testLetters.size(),478,CV_32F);
            NormResample(iniQueryVector, testLetters,Letters); //STILL REVIEW FOR DFT
            ExtractDFTCoeff(Letters,testData);
            cv::FileStorage file2("testData_norm.yml", cv::FileStorage::WRITE);

            // Write to file!
            file2 << "Test Data" << testData;
            file2.release();
            CvMat* sample = cvCreateMat(1,480,CV_32FC1);
            *sample = testData.clone();
            int k = 7;
            CvMat* neighborResponses = cvCreateMat(1,k,CV_32FC1);
            CvMat* dist = cvCreateMat(1,k,CV_32FC1);
            CvMat* results = cvCreateMat(1,1,CV_32FC1);
            float detectedClass = knearest.find_nearest(sample,k,results,0,neighborResponses,dist);
            cout << "The number you wrote is this: " << (int) (detectedClass) << endl;
//            finger_2d.clear();
            cv::Mat nneighResp = cv::Mat(neighborResponses,true);
            cv::Mat ddist = cv::Mat(dist,true);
            cv::Mat rresults = cv::Mat(results,true);

            cv::FileStorage file3("neighborResponses.yml", cv::FileStorage::WRITE);

            // Write to file!
            file3 << "Neighbor Responses" << nneighResp;
            file3.release();

            cv::FileStorage file4("distances.yml", cv::FileStorage::WRITE);

            // Write to file!
            file4 << "Distances" << ddist;
            file4.release();

            cv::FileStorage file5("results.yml", cv::FileStorage::WRITE);

            // Write to file!
            file5 << "Results" << rresults;
            file5.release();

            number.clear();
            array_counter = 0;

            state = 0;
        }

        if (state == 0) {
        std::cout << "Press 's' to Start or Stop recording" << std::endl
          << "Press 'n' to go back to main menu, and other keys to exit." << std::endl;
        }
        std::cin >> c;
        }
    }
    // Remove the sample listener when done
    controller.removeListener(listener);
    output_file.close();

    /*//Comenzar a extraer las submatrices
    int correct(0), incorrect(0);
    cout << "////////////////////////////////////" << endl;
    cout << "         Start testing KNN          " << endl;
    cout << "////////////////////////////////////" << endl;
    for (int i=1;i<oldLetters.size()-1;i++) {
    //Let's start with just one
//    int i=20;
    cv::Mat trainD = cv::Mat::eye(nof-1,478,CV_32F); //THIS SIZE SHOULD BE MODIFIED MANUALLY
    cv::Mat testD = cv::Mat::eye(1,478,CV_32F);
    cv::Mat trainL = cv::Mat::eye(nof-1,1,CV_32F);
    cv::Mat testL = cv::Mat::eye(1,1,CV_32F);
    //Copy for data matrix
    cv::Size smallsize1(478,i);
    cv::Size smallsize2(478,nof-i-1);
    cv::Rect rectD1 = cv::Rect(0,0,smallsize1.width,smallsize1.height);
    cv::Rect rectD2 = cv::Rect(0,smallsize1.height+1,smallsize2.width,smallsize2.height);
    cv::Rect rectD3 = cv::Rect(0,smallsize1.height,smallsize2.width,1);
    cv::Mat digD1 = cv::Mat(trainData,rectD1);
    cv::Mat digD2 = cv::Mat(trainData,rectD2);
    cv::Mat digD3 = cv::Mat(trainData,rectD3);
    digD1.copyTo(trainD(cv::Rect(0,0,smallsize1.width,smallsize1.height)));
    digD2.copyTo(trainD(cv::Rect(0,smallsize1.height,smallsize2.width,smallsize2.height)));
    digD3.copyTo(testD);
    //Copy for test matrix
    cv::Size smallsizeT1(1,i);
    cv::Size smallsizeT2(1,nof-i-1);
    cv::Rect rectT1 = cv::Rect(0,0,smallsizeT1.width,smallsizeT1.height);
    cv::Rect rectT2 = cv::Rect(0,smallsizeT1.height+1,smallsizeT2.width,smallsizeT2.height);
    cv::Rect rectT3 = cv::Rect(0,smallsizeT1.height,smallsizeT2.width,1);
    cv::Mat digT1 = cv::Mat(trainClasses,rectT1);
    cv::Mat digT2 = cv::Mat(trainClasses,rectT2);
    cv::Mat digT3 = cv::Mat(trainClasses,rectT3);
    digT1.copyTo(trainL(cv::Rect(0,0,smallsizeT1.width,smallsizeT1.height)));
    digT2.copyTo(trainL(cv::Rect(0,smallsizeT1.height,smallsizeT2.width,smallsizeT2.height)));
    digT3.copyTo(testL);
    CvMat ttrainData = trainD;
    CvMat ttrainClasses = trainL;
    cv::KNearest knearest(&ttrainData,&ttrainClasses);

    CvMat* sample = cvCreateMat(1,478,CV_32FC1);
    *sample = testD.clone();
    float detectedClass = knearest.find_nearest(sample,1);
    int z((int)testL.at<float>(0,0));
    if (z != (int) ((detectedClass)))
        {
            cout << "False. It's " << z << " but it gave me " << (int) ((detectedClass)) << endl;
            incorrect++;
//                exit(1);
        }
    if (z == (int) ((detectedClass)))
        {
            cout << "Correct! " << (int) ((detectedClass)) << endl;
            correct++;
        }
    }
    cout << "Number of correct guesses = " << correct << endl;
    cout << "Number of incorrect guesses = " << incorrect << endl;


    // Declare what you need
    cv::FileStorage file("trainClasses_norm.yml", cv::FileStorage::WRITE);
//    cv::FileStorage file("trainData_norm.yml", cv::FileStorage::WRITE);

    // Write to file!
    file << "Train Classes" << trainClasses;
//    file << "Train Data" << trainData;
    file.release();*/


    return 0;
}

/*vector<vector<float>> ExtractDFTCoeff(vector<vector<float>> &new_character)
{
//    vector<complex<float>> complex_shape, complex_dft;
//    for (int i=0;i<new_character.size();i++)
//    {
//        complex<float> xiy(new_character[i][0],new_character[i][1]);
//        complex_shape.push_back(xiy);
//    }
    int m(new_character.size());
    int i,k;
    float arg, cosarg, sinarg;
    float tempx, tempy;
//    vector<float> x2,y2;
    vector<float> cmplx;
    vector<vector<float>> dft;
//    x2.reserve(new_character[0].size());
//    y2.reserve(new_character[0].size());
    for(i=0;i<(int)(0.8*m);i++) // Keep only 80% of the coeff
//    for(i=0;i<m;i++)
    {
        cmplx.clear();
        tempx = 0;
        tempy = 0;
        arg = -1 * 2.0 * 3.141592654 * (float)i / (float)m;
        for (k=0;k<m;k++)
        {
            cosarg = cos(k*arg);
            sinarg = sin(k*arg);
            tempx += (new_character[k][0] * cosarg - new_character[k][1] * sinarg);
            tempy += (new_character[k][0] * sinarg + new_character[k][1] * cosarg);
        }
        if (i==0) cout<< "Sum of values = " << tempx << " , " << tempy << endl;
        cmplx.push_back(tempx); cmplx.push_back(tempy);
        dft.push_back(cmplx);
//        x2.push_back(tempx);
//        y2.push_back(tempy);
    }
    return dft;
}*/

void ExtractDFTCoeff(vector<vector<vector<float>>> &new_characters, cv::Mat& trainData)
{
    int m(new_characters[0].size()); //cout << "Valor de m: " << m << endl;
    int i,j,k;
    int new_m;
    float arg, cosarg, sinarg;
    float tempx, tempy;
    
    for (j=0;j<new_characters.size();j++)
    {
        float scaleInv(0.5), fraction(0.8);
        vector<float> cmplx;
        vector<vector<float>> dft(m, vector<float>(2,0));
        vector<vector<float>> dft_shift(m, vector<float>(2,0));
        vector<vector<float>> dft_extract((int)(m*fraction), vector<float>(2,0));
        vector<vector<float>> dft_ishift((int)(m*fraction), vector<float>(2,0));
        for(i=0;i<m;i++) 
        {
            cmplx.clear();
            tempx = 0;
            tempy = 0;
            arg = -1 * 2.0 * 3.141592654 * (float)i / (float)m;
            for (k=0;k<m;k++)
            {
                cosarg = cos(k*arg);
                sinarg = sin(k*arg);
                tempx += (new_characters[j][k][0] * cosarg - new_characters[j][k][1] * sinarg);
                tempy += (new_characters[j][k][0] * sinarg + new_characters[j][k][1] * cosarg);
            }
            tempx = tempx/m;
            tempy = tempy/m;
            if(i==1) scaleInv = sqrt(tempx*tempx + tempy*tempy);
            tempx = tempx/scaleInv;
            tempy = tempy/scaleInv;
            dft[i][0] = tempx; dft[i][1] = tempy; 
            // The following if is implementing the function fftshift doing a circular shift
            if (i < m/2) { dft_shift[i+(int)(m/2)][0] = tempx; dft_shift[i+(int)(m/2)][1] = tempy; }
            else { dft_shift[i-(int)(m/2)][0] = tempx; dft_shift[i-(int)(m/2)][1] = tempy; }
        }

        //Now from dft_shift we extract a fraction of the coeff (Filter part)
        //For some reason when doing (int) it gets rounded one less, we need to add one more
        for (i=(int)(m*(1-fraction)*0.5+1);i<(int)(m*(1+fraction)*0.5);i++)
        {
            dft_extract[i-(int)(m*(1-fraction)*0.5+1)][0] = dft_shift[i][0];
            dft_extract[i-(int)(m*(1-fraction)*0.5+1)][1] = dft_shift[i][1];
        }

        //Next we do the function ifftshift doing again a circular shift
        new_m = dft_extract.size();
        for (i=0;i<new_m;i++)
        {
            if (i < new_m/2) { dft_ishift[i+(int)(new_m/2)][0] = dft_extract[i][0]; dft_shift[i+(int)(new_m/2)][1] = dft_extract[i][1]; }
            else { dft_ishift[i-(int)(new_m/2)][0] = dft_extract[i][0]; dft_shift[i-(int)(new_m/2)][1] = dft_extract[i][1]; }
        }
        //Finally we drop the frequency coeff at 0 to make our Fourier descriptor translation invariant
        dft_ishift.erase(dft_ishift.begin()); //Remember that our vector is 1 element less now
        //Now we need to store them in the trainData matrix
        for (i=0;i<dft_ishift.size();i++)
        {
            trainData.at<float>(j,i) = dft_ishift[i][0]; trainData.at<float>(j,dft_ishift.size()+i) = dft_ishift[i][1];
        }
    }
//    cout << "ALL GOOD" << endl;
}

void NormResample(vector<vector<float> > &iniQueryVector, vector<vector<vector<float>>> &oldLetters, vector<vector<vector<float>>> &newLetters)
{
    //First step calculate mean and variance for each character in x and y
    vector<vector<float>> norm_character;
    vector<float> xelem, yelem, newx, newy;
    //Create new query for interpolation 0:1/(N-1):1
    int N=300; //Number of new query points
    vector<float> newquery(300);
    iota(begin(newquery),end(newquery),0);
    transform(newquery.begin(),newquery.end(),newquery.begin(),bind2nd(divides<float>(),(N-1)));
//    cout << "Value of the new query vector: " << newquery[20] << endl;
    //Normalize every set of data
    for (int i=0;i<oldLetters.size();i++)
    {
        ExtractXYcomp(oldLetters[i], xelem, yelem);
        // Calculate mean
        float sumx = accumulate(xelem.begin(), xelem.end(), 0.0);
        float meanx = sumx / xelem.size();
        float sumy = accumulate(yelem.begin(), yelem.end(), 0.0);
        float meany = sumy / yelem.size();
        // Calculate std. deviation
        vector<float> diffx(xelem.size());
        transform(xelem.begin(), xelem.end(), diffx.begin(),
                       bind2nd(minus<float>(), meanx));
        float sq_sumx = inner_product(diffx.begin(), diffx.end(), diffx.begin(), 0.0);
        float stdx = sqrt(sq_sumx / xelem.size());
        vector<float> diffy(yelem.size());
        transform(yelem.begin(), yelem.end(), diffy.begin(),
                       bind2nd(minus<float>(), meany));
        float sq_sumy = inner_product(diffy.begin(), diffy.end(), diffy.begin(), 0.0);
        float stdy = sqrt(sq_sumy / yelem.size());
        // Normalize xlem and yelem as x_new = (x-mean)/std
        transform(xelem.begin(),xelem.end(),xelem.begin(),bind2nd(minus<float>(),meanx));
        transform(xelem.begin(),xelem.end(),xelem.begin(),bind2nd(divides<float>(),stdx));
        transform(yelem.begin(),yelem.end(),yelem.begin(),bind2nd(minus<float>(),meany));
        transform(yelem.begin(),yelem.end(),yelem.begin(),bind2nd(divides<float>(),stdy));
        transform(iniQueryVector[i].begin(),iniQueryVector[i].end(),iniQueryVector[i].begin(),bind2nd(divides<float>(),(iniQueryVector[i].size()-1)));
        newx = interp1(iniQueryVector[i],xelem,newquery);
        newy = interp1(iniQueryVector[i],yelem,newquery);
        InsertXYcomp(newx,newy,norm_character); //New output will be 2xN
        newLetters.push_back(norm_character);
        norm_character.clear(); xelem.clear(); yelem.clear(); newx.clear(); newy.clear();
    }
}

void ExtractXYcomp(vector<vector<float>> &character, vector<float> &compx, vector<float> &compy)
{
    for (int i=0;i<character.size();i++)
    {
        compx.push_back(character[i][0]);
        compy.push_back(character[i][1]);
    }
}

void InsertXYcomp(vector<float> &compx, vector<float> &compy, vector<vector<float>> &new_character)
{
    vector<float> temp;
    for (int i=0;i<compx.size();i++)
    {
        temp.push_back(compx[i]); temp.push_back(compy[i]);
        new_character.push_back(temp);
        temp.clear();
    }
//    new_character.push_back(compx);
//    new_character.push_back(compy);
}

vector< float > interp1( vector< float > &x, vector< float > &y, vector< float > &x_new )
{
    vector< float > y_new;
    y_new.reserve( x_new.size() );

    std::vector< float > dx, dy, slope, intercept;
    dx.reserve( x.size() );
    dy.reserve( x.size() );
    slope.reserve( x.size() );
    intercept.reserve( x.size() );
    for( int i = 0; i < x.size(); ++i ){
        if( i < x.size()-1 )
        {
            dx.push_back( x[i+1] - x[i] );
            dy.push_back( y[i+1] - y[i] );
            slope.push_back( dy[i] / dx[i] );
            intercept.push_back( y[i] - x[i] * slope[i] );
        }
        else
        {
            dx.push_back( dx[i-1] );
            dy.push_back( dy[i-1] );
            slope.push_back( slope[i-1] );
            intercept.push_back( intercept[i-1] );
        }
    }
    for ( int i = 0; i < x_new.size(); ++i )
    {
        int idx = findNearestNeighbourIndex( x_new[i], x );
        y_new.push_back( slope[idx] * x_new[i] + intercept[idx] );
    }
    return y_new;
}

int findNearestNeighbourIndex( float value, vector< float > &x )
{
    float dist = FLT_MAX;
    int idx = -1;
    for ( int i = 0; i < x.size(); ++i ) {
        float newDist = value - x[i];
        if ( newDist > 0 && newDist < dist ) {
            dist = newDist;
            idx = i;
        }
    }

    return idx;
}

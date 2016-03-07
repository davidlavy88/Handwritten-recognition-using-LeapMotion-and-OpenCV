# Handwritten-recognition-using-LeapMotion-and-OpenCV
This is my final project in my Image Processing. Data acquired by Leap Motion will be classified using KNN search.
Stage 1: Using it in existing data (file digits.png from opencv/samples/python2/data)
Code was based on: 
1. http://docs.opencv.org/ref/master/d8/d4b/tutorial_py_knn_opencv.html
2. http://my-tech-talk.blogspot.com/2012/06/digit-recognition-with-opencv.html
Stage 2: Extract data from users as training set. Runs offline version
Stage 3: Run online version, extracting test data in real time

Files:

test_dft: Extracts DFT coefficients for the data extracted from Leap.
knn_OpenCV: Example of how to use knn using the OpenCV API using random generated 2D points.
test_knn: Test the KNN algorithm (from OpenCV) using the handwritten digits from the opencv/samples/python2/data folder file.
main_LOOCV: This program runs an offline version of the project. By selecting some of the numbers and store them in the folder called 2D_Data with different name format (check the folder and the cpp file) it trains a kNN classification system and calculates the performance using the LOOCV approach (Leave One Out Cross-Validation)
main: This program runs the online version of the project. Uses the training set with the same different format as the offline version. Once it's trained it will record data from the Leap and classify it. For checking the metrics of the calculations, extra files has been added in yml format.

Folder:

Final Numbers: Training set from 12 different people for all the numbers
2D_Data: Samples from the training set with the required format for some of the cpp files


To compile, extract in a folder (e.g leapknn_first) and run: (Number)(Random_Letter)_2d.txt

$ cmake .
$ make
$ ./leapknn_rt

#include "UltraFace.hpp"
#include <iostream>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
int main() {


    string bin_path = "slim_320.bin";
    string param_path = "slim_320.param";
    
    VideoCapture cap;
    cap.open(0);

    if (!cap.isOpened())
    {
        cout << "camera failed!" << "\n";
        return -1;
    }

    Mat img;
    while (true)
    {   
        Mat frame;
        cap>>frame;
        if (frame.empty()) return 0;
        UltraFace ultraface(bin_path, param_path, 320, 240, 1, 0.7);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            Point pt1(face.x1, face.y1);
            Point pt2(face.x2, face.y2);
            rectangle(frame, pt1, pt2, Scalar(0, 0, 255), 1);
        }
        imshow("Video", frame);
        if (waitKey(30) >= 0) break;
    }
    cap.release();
    return 0;
    /*UltraFace ultraface(bin_path, param_path, 320, 240, 1, 0.7); // config model input

    for (int i = 3; i < argc; i++) {
        string image_file = argv[i];
        cout << "Processing " << image_file << endl;

        Mat frame = imread(image_file);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            Point pt1(face.x1, face.y1);
            Point pt2(face.x2, face.y2);
            rectangle(frame, pt1, pt2, Scalar(0, 0, 255), 1);
        }

        imshow("UltraFace", frame);
        waitKey();
        imwrite("result.jpg", frame);
    }*/
}

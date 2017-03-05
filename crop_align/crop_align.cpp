//
// Created by szhou on 11/26/16.
//
#include <iostream>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include "imtransform.h"

using namespace std;

int main(int argc, char** argv) {
    if (argc != 11) {
        cerr << "Usage: " << argv[0] << " image_path save_path ecx ecy ulx uly target_width target_height target_ecy target_uly" << endl;
        return 1;
    }
    string file_path = argv[1];
    string save_path = argv[2];
    float ecx = stof(argv[3]);
    float ecy = stof(argv[4]);
    float ulx = stof(argv[5]);
    float uly = stof(argv[6]);
    // new version
    int tar_w = stoi(argv[7]);
    int tar_h = stoi(argv[8]);
    float tar_ecy = stof(argv[9]);
    float tar_uly = stof(argv[10]);

    cv::Mat img = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);

    cv::Mat transformed = sim_transform_image_3channels(img, tar_w, tar_h, ecx, ecy, ulx, uly, tar_ecy, tar_uly);
    cv::imwrite(save_path, transformed);
    return 0;
}
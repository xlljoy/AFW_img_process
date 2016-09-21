//
//  face_crop.hpp
//  afw_face_crop
//
//  Created by LiXile  on 2016-09-16.
//  Copyright © 2016 LiXile . All rights reserved.
//

#ifndef face_crop_hpp
#define face_crop_hpp


#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class convert2lmdb
{
    
    vector<vector<int>> afw_face_num_;
    vector<vector<Rect>> afw_face_rec_;
    vector<vector<string>> afw_face_path_;
    vector<vector<Mat>>afw_face_img;
    
    
    
    vector<int> Num_face_;
    vector<Rect> Rect_face_;
    vector<Mat> face_small_;
    vector<string> Path_full_;
    vector<Mat> images_;
    vector<string>mid_file_list_;

    
public:
    
    void get_mid_file_list(string mid_file_list_txt, string headpath);
    void rec_read(string headpath);
    void num_read(string headpath);
    void img_path_read(string headpath);
    vector<Mat> img_Mat_read(string headpath, int i);
    float IoM(Rect rect_1, Rect rect_2);
    int get_nonface_dimension(vector<Rect> rects);
    vector<Rect> get_nonface_rect(Mat img, int width, vector<Rect>rects, float threshold);
    void txt2jpg(string headpath, string headpath_img, string mid_file_list_txt, string smallface_output_path, float threshold, string nonface_output_path);
    
    
    
    
    void drawfaceRect(Mat face, vector<Rect> rect);
    
    void txtRead();
    void txtRead(string head_path, string smallface_output_path, string mid_file_list_txt);
        string convert_path(string head_path,string mid_file_list, string path);
    Rect convert_rect(string path);
    int convert_num(string path);
    Mat readImg(string path);
    Mat get_smallface(Mat img, Rect_<float> rect);
    void translation();
    void convert_JPG(Mat img, int num, string smallface_output_path);
    void JPG_txt(string smallface_from,string smallface_output_path);
    void JPG_datum(string path);
    void datum_lmdb(string path);
    void txt_lmdb(string head_path, string smallface_output_path,string mid_file_list_txt);
};
#endif /* convert2lmdb_hpp */

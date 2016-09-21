//
//  main.cpp
//  caffe_trainingdata0827
//
//  Created by LiXile  on 2016-08-27.
//  Copyright © 2016 LiXile . All rights reserved.
//

#include <iostream>
#include "face_crop.hpp"

int main() {
    convert2lmdb test;
    string head_path = "/Users/lixile/Downloads/v1/";
    string mid_file_list_txt = "name_list.txt";
    string headpath_img = "/Users/lixile/Downloads/WIDER_train/images/";
    string smallface_output_path = "/Users/lixile/Downloads/WIDER_train/small_face/";
    string small_nonface_output_path = "/Users/lixile/Downloads/WIDER_train/small_nonface/";
    int threshold = 0.1;
    //test.txt_lmdb("/Users/lixile/Downloads/WIDER_train/images/",  "/Users/lixile/Documents/Programming/AFW_face_crop_0916", "/Users/lixile/Documents/Programming/AFW_face_crop_0916/name_list.txt");
//    test.get_mid_file_list(mid_file_list_txt, head_path);
//    test.num_read(head_path);
//    test.rec_read(head_path);
//    test.img_path_read(head_path);
    //test.img_Mat_read(headpath_img);
    test.txt2jpg(head_path, headpath_img,  mid_file_list_txt, smallface_output_path, threshold, small_nonface_output_path);
    int i=0;
//    Rect a(1, 2, 50, 50);
//    Rect b(35, 20, 80, 50);
//    vector<Rect> c;
//    c.push_back(a);
//    c.push_back(b);
//    int d = test.get_nonface_dimension(c);
//    float m = test.IoM(a, b);
    
    return 0;
}

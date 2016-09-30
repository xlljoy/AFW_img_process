//
// Created by xileli on 9/27/16.
//
#include "AFW_face_crop.h"

int main()
{
    convert2lmdb test;
    std::string head_path = "/home/xileli/Documents/dateset/AFW/v1/";
    std::string mid_file_list_text = "name_list.txt";
    std::string headpath_img = "/home/xileli/Documents/dateset/AFW/WIDER_train/images/";
    std::string smallface_output_path = "/home/xileli/Documents/dateset/AFW/WIDER_train/small_face/";
    std::string small_nonface_output_path = "/home/xileli/Documents/dateset/AFW/WIDER_train/small_nonface/";
    int threshold = 0.1;
    test.txt2jpg(head_path, headpath_img, mid_file_list_text, small_nonface_output_path, threshold, small_nonface_output_path);
    return 0;
 }


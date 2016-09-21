//
//  face_crop.cpp
//  afw_face_crop
//
//  Created by LiXile  on 2016-09-16.
//  Copyright © 2016 LiXile . All rights reserved.
//

#include "face_crop.hpp"
//
//  convert2lmdb.cpp
//  caffe_trainingdata0827
//
//  Created by LiXile  on 2016-08-27.
//  Copyright © 2016 LiXile . All rights reserved.
//





void convert2lmdb::rec_read(string headpath)
{
    for(int i=0; i<mid_file_list_.size(); i++)
    {
    string rec_read_path = headpath + mid_file_list_[i] + ".txt";
    vector<Rect> cur_rects;
    string line;
    Rect cur_rect;
    ifstream rec_read(rec_read_path);
    while(!rec_read.eof())
    {
        getline(rec_read, line);
        if(line == "")continue;
        cur_rect = convert_rect(line);
        cur_rects.push_back(cur_rect);
    }
    afw_face_rec_.push_back(cur_rects);
    }
}

void convert2lmdb::num_read(string headpath)
{
    for(int i=0; i<mid_file_list_.size(); i++)
    {
        string num_read_path = headpath + mid_file_list_[i] + "_head_num.txt";
        vector<int> cur_nums;
        string line;
        int cur_num;
        ifstream num_read(num_read_path);
        while(!num_read.eof())
        {
            getline(num_read, line);
            if(line == "")continue;
            cur_num = convert_num(line);
            cur_nums.push_back(cur_num);
            
        }
        afw_face_num_.push_back(cur_nums);
    }
}


void convert2lmdb::img_path_read(string headpath)
{
    for(int i=0; i<mid_file_list_.size(); i++)
    {
        string img_read_path = headpath + mid_file_list_[i] + "_img_path.txt";
        vector<string> cur_img_pathes;
        string line;
        ifstream img_path_read(img_read_path);
        while(!img_path_read.eof())
        {
            getline(img_path_read, line);
            if(line == "")continue;
            cur_img_pathes.push_back(line);
        }
        afw_face_path_.push_back(cur_img_pathes);
    }
   
}


vector<Mat> convert2lmdb::img_Mat_read(string headpath_img, int i)
{
    
        vector<Mat> cur_imgs;
        Mat cur_img;
        for(int j=0; j<afw_face_path_[i].size(); j++)
        {
            string img_path = headpath_img + mid_file_list_[i] + afw_face_path_[i][j];
            cur_img = imread(img_path);
            cur_imgs.push_back(cur_img);
        }
    return cur_imgs;
}




float convert2lmdb::IoM(Rect rect_1, Rect rect_2)
{
    int x11 = rect_1.x;
    int y11 = rect_1.y;
    int x12 = rect_1.width+x11;
    int y12 = rect_1.height+y11;
    int x21 = rect_2.x;
    int y21 = rect_2.y;
    int x22 = rect_2.width+x21;
    int y22 = rect_2.height+y22;
    int x_overlap = std::max(0, (std::min(x12, x22) - std::max(x11, x21)));
    int y_overlap = max(0, min(y12, y22) - max(y11, y21));
    int intersection = x_overlap * y_overlap;
    int rect_1_area = (y12 - y11) * (x12 -x11);
    int rect_2_area = (y22 - y21) * (x22 - x21);
    int min_area = min(rect_1_area, rect_2_area);
    float result = intersection * 1.0 /min_area;
    return result;
}

int convert2lmdb::get_nonface_dimension(vector<Rect>rects)
{
    int dimension =  rects[0].width;
    for(int i=0; i<rects.size(); i++)
    {
        if(dimension < rects[i].width)
            dimension = rects[i].width;
    }
    return dimension;
}


vector<Rect> convert2lmdb::get_nonface_rect(Mat img, int width, vector<Rect>rects, float threshold)
{
    vector<Rect> nonface_rects;
    for(int i=0; i<img.cols-width; i+=width)
    {
        for(int j=0; j<img.rows-width; j+=width)
        {
            Rect cur_rect(i, j, width, width);
            for(int k=0; k<rects.size(); k++)
            {
                float overlap = IoM(rects[k],cur_rect);
                if(overlap <= threshold)
                    nonface_rects.push_back(cur_rect);
            }
        }
    }
    return nonface_rects;
}


void convert2lmdb::txt2jpg(string headpath, string headpath_img, string mid_file_list_txt, string smallface_output_path, float threshold, string nonface_output_path)
{
    get_mid_file_list(mid_file_list_txt, headpath);
    num_read(headpath);
    rec_read(headpath);
    img_path_read(headpath);
    int64 num = 0, non_num=0;
    ofstream smallface_file, small_nonface_file;
    smallface_file.open(smallface_output_path+"face_cropped.txt");
    //small_nonface_file.open(nonface_output_path+"nonface_cropped.txt");
    for(int i=0; i<mid_file_list_.size(); i++)
    {
        vector<Mat> cur_imgs;
        cur_imgs = img_Mat_read(headpath_img, i);
        
        int count = 0;
        int img_count = cur_imgs.size();
        for(int j = 0; j < img_count; j++)
        {
            vector<Rect> cur_face_rects, nonface_rects;
            for(int k=0; k<afw_face_num_[i][j]; k++)
            {
                
                cur_face_rects.push_back(afw_face_rec_[i][count]);

            Mat small_face = get_smallface( cur_imgs[j], afw_face_rec_[i][count]);
            convert_JPG(small_face, num, smallface_output_path);
            
            smallface_file<<"/Users/lixile/Downloads/WIDER_train/small_face/"<<num<<".jpg"<<' '<<1<<endl;
            num++;
            count++;
            }
            int dimension = get_nonface_dimension(cur_face_rects);
            nonface_rects = get_nonface_rect(cur_imgs[j], dimension, cur_face_rects, threshold);
            
            for(int l=0; l<nonface_rects.size(); l++)
            {
                Mat small_nonface = get_smallface( cur_imgs[j], nonface_rects[l]);
                convert_JPG(small_nonface, num, nonface_output_path);
                smallface_file<<"/Users/lixile/Downloads/WIDER_train/small_nonface/"<<num<<".jpg"<<' '<<0<<endl;
                num++;
            }
            
            
        }
    }
   
}



void convert2lmdb::drawfaceRect(Mat face, vector<Rect> rect)
{
    for(int i = 0; i < rect.size(); i++)
    {
        rectangle(face, rect[i], Scalar(1,0,0));
    }
    imshow("test", face);
    waitKey(0);
}











void convert2lmdb::txtRead(string head_path, string smallface_output_path, string mid_file_list_txt)
{
   
    get_mid_file_list(mid_file_list_txt, head_path);
    num_read(head_path);
    rec_read(head_path);
    img_path_read(head_path);
    ifstream mid(mid_file_list_txt);
    ofstream smallface_file;
    string line, mid_file_list, Num_face_path,headnum,Rec_path,rec_line, img_path,txt_path;
    int count=0,num=0;
    cv::Mat Img,Img_small;
    txt_path = "/Users/lixile/Downloads/v1/";
    smallface_file.open(smallface_output_path+"face_cropped.txt");
    //    smallface_file.open(smallface_output_path+"1.csv");
    while (!mid.eof())
    {
        getline(mid, mid_file_list);
        img_path = txt_path + mid_file_list + "_img_path.txt";
        //string img_path2 = "/Users/lixile/Downloads/v1/0--Parade_img_path.txt";
        ifstream in(img_path);
        while(!in.eof())
        {
            getline(in, line);
            getline(in, line);
            Img=readImg(convert_path(head_path,mid_file_list,line));
            //string m = convert_path(head_path,mid_file_list,line);
            images_.push_back(Img);
            Path_full_.push_back(convert_path(head_path,mid_file_list, line));
        
   
        Num_face_path = txt_path+mid_file_list+"_head_num.txt";
        Rec_path = txt_path + mid_file_list+".txt";
        ifstream head_num(Num_face_path);
        ifstream rec_read(Rec_path);
        while(!head_num.eof())
        {
            getline(head_num, headnum);
            count=convert_num(headnum);
            Num_face_.push_back(count);
            
            for(int j=0;j<count;j++)
            {
                
                getline(rec_read, rec_line);
                Rect_face_.push_back(convert_rect(rec_line));
                Img_small=get_smallface(Img, convert_rect(rec_line));
                face_small_.push_back(Img_small);
                convert_JPG(Img_small, num, smallface_output_path);
                //                smallface_file<<smallface_from<<","<<smallface_output_path<<num<<".jpg"<<","<<1<<endl;
                smallface_file<<num<<".jpg"<<' '<<1<<endl;
                num++;
            }
        }
        }
    }
    
}


void convert2lmdb::get_mid_file_list(string mid_file_list_txt,string headpath)
{
    ifstream in(headpath+mid_file_list_txt);
    string line;
    while(!in.eof())
    {
        getline(in, line);
        if(line == "")continue;
        mid_file_list_.push_back(line);
    }
}


string convert2lmdb::convert_path(string head_path,string mid_file_list, string path)
{
    return head_path+mid_file_list+path;
}

Rect convert2lmdb::convert_rect(string path)
{
    char *cstr=new char[path.length()+1];
    strcpy(cstr,path.c_str());
    char *p=strtok(cstr, ",");
    int i=0;
    vector<float> rect_list;
    while(p!=0)
    {
        
            float pos= atof(p);
            rect_list.push_back(pos);
            i++;
            p=strtok(NULL,",");
    }
    //ellipse_rect(rect_list[0], rect_list[1], rect_list[2], rect_list[3]);
    return Rect(rect_list[0], rect_list[1], rect_list[2], rect_list[3]);
    
    
}



int convert2lmdb::convert_num(string path)
{
    return atoi(path.c_str());
}

cv::Mat convert2lmdb::readImg(string path)
{
    return cv::imread(path);
}

cv::Mat convert2lmdb::get_smallface(cv::Mat img, cv::Rect_<float> rect)
{
    if(rect.x<=0) rect.x=0;
    if(rect.y<=0) rect.y=0;
    if(img.cols<(rect.x+rect.width)) rect.width=img.cols-rect.x;
    if(img.rows<(rect.y+rect.height)) rect.height=img.rows-rect.y;
    return img(rect);
}

void convert2lmdb::convert_JPG(cv::Mat img, int num, string smallface_output_path)
{
    string path(smallface_output_path);
    string n=to_string(num);
    cv::imwrite(path+n+".jpg", img);
}


void convert2lmdb::JPG_datum(string path){}
void convert2lmdb::datum_lmdb(string path){}
void convert2lmdb::txt_lmdb( string head_path, string smallface_output_path, string mid_file_list_txt)
{
    txtRead( head_path,  smallface_output_path , mid_file_list_txt);
    
}

void translation()
{
    
}



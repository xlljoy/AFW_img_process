cmake_minimum_required(VERSION 3.6)
project(AFW_img_process)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(OpenCV_DIR "/usr/local/lib")
find_package(OpenCV)

set(SOURCE_FILES main.cpp AFW_face_crop.cpp AFW_face_crop.h)
add_executable(AFW_img_process ${SOURCE_FILES} AFW_face_crop.cpp AFW_face_crop.h)
target_link_libraries(AFW_img_process ${OpenCV_LIBS})
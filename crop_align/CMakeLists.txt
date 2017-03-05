cmake_minimum_required(VERSION 2.8)
project(ImageCropAlign)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
find_package( OpenCV )
MESSAGE( STATUS "OpenCV:         " ${OpenCV_INCLUDE_DIRS} )

add_executable(crop_align crop_align.cpp imtransform.cpp)
target_link_libraries(crop_align ${OpenCV_LIBS})

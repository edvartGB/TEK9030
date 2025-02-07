cmake_minimum_required(VERSION 3.10)
project(lab_segmentation)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(lab_segmentation
    ${CMAKE_SOURCE_DIR}/include/common-lab-utils.h
    ${CMAKE_SOURCE_DIR}/include/common-lab-utils.cpp
    ${CMAKE_SOURCE_DIR}/include/lab-segmentation.h
    ${CMAKE_SOURCE_DIR}/include/lab-segmentation.cpp
    ${CMAKE_SOURCE_DIR}/include/multivariate-normal-model.h
    ${CMAKE_SOURCE_DIR}/include/multivariate-normal-model.cpp
    main.cpp
)

target_link_libraries(lab_segmentation ${OpenCV_LIBS})

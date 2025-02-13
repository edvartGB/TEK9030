cmake_minimum_required(VERSION 3.10)
project(lab_feature_detection)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(lab_feature_detection
    ${CMAKE_SOURCE_DIR}/include/circle_estimator.cpp
    ${CMAKE_SOURCE_DIR}/include/circle_estimator.h
    ${CMAKE_SOURCE_DIR}/include/circle.cpp
    ${CMAKE_SOURCE_DIR}/include/circle.h
    ${CMAKE_SOURCE_DIR}/include/corner_detector.cpp
    ${CMAKE_SOURCE_DIR}/include/corner_detector.h
    ${CMAKE_SOURCE_DIR}/include/filters.cpp
    ${CMAKE_SOURCE_DIR}/include/filters.h
    ${CMAKE_SOURCE_DIR}/include/lab_corners.cpp
    ${CMAKE_SOURCE_DIR}/include/lab_corners.h
    main.cpp
)

target_link_libraries(lab_feature_detection ${OpenCV_LIBS})

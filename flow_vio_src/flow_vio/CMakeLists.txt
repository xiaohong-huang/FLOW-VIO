cmake_minimum_required(VERSION 2.8.3)
project(flow_vio)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport
    rosbag)
    
    
# find_package(OpenCV 4 REQUIRED)
include(../../opencv/build/OpenCVConfig.cmake)
find_package(Eigen3)
find_package ( OpenMP )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  src/swf/
  src/factor/
  src/utility/
  src/initial/
  src/feature/
)

catkin_package()
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    catkin libraries: ${catkin_LIBRARIES}")

add_library(flow_vio_lib
    src/parameter/parameters.cpp
    src/swf/swf.cpp
    src/swf/swf_imu.cpp
    src/swf/swf_image.cpp
    src/swf/swf_core.cpp
    src/feature/feature_manager.cpp
    src/feature/feature_tracker.cpp
    src/factor/marginalization_factor.cpp
    src/factor/imu_factor.cpp
    src/factor/integration_base.cpp
    src/factor/initial_factor.cpp
    src/factor/visual_inerial_factor.cpp
    src/factor/loss_function.cpp
    src/factor/visual_inerial_base.cpp
    src/factor/visual_inerial_base_trans.cpp
    src/factor/visual_inerial_base_rhs.cpp
    src/solver/solver.cpp
    src/solver/solver_residualblock.cpp
    src/utility/visualization.cpp
    src/utility/camera_pose_visualization.cpp
    
    )
    
target_link_libraries(flow_vio_lib ${OpenCV_LIBS} ${catkin_LIBRARIES} )


add_executable(flow_vio_node src/main3.cpp)
target_link_libraries(flow_vio_node flow_vio_lib) 

add_executable(4seasons_convert2bag_node src/4seasons_convert2bag.cpp)
target_link_libraries(4seasons_convert2bag_node flow_vio_lib) 


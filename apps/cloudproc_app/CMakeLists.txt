add_executable(cloudproc_app main.cpp)
target_link_libraries(cloudproc_app common_lib)
target_link_libraries(cloudproc_app dbg_visualizer_lib)
target_link_libraries(cloudproc_app imgproc_lib)
target_link_libraries(cloudproc_app cloudproc_lib)
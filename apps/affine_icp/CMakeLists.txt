add_executable(affine_icp_app main.cpp)
target_link_libraries(affine_icp_app common_lib)
target_link_libraries(affine_icp_app dbg_visualizer_lib)
target_link_libraries(affine_icp_app imgproc_lib)
target_link_libraries(affine_icp_app cloudproc_lib)
target_link_libraries(affine_icp_app affine_kinematic_lib)
target_link_libraries(affine_icp_app rigid_kinematic_lib)
target_link_libraries(affine_icp_app nn_search_lib)
target_link_libraries(affine_icp_app gmm_corr_lib)
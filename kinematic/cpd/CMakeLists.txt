file(GLOB CPD_KINEMATIC_SRC *.h *.hpp *.cpp *.cu)


#Add the library
cuda_add_library(cpd_kinematic_lib ${CPD_KINEMATIC_SRC})
target_link_libraries(cpd_kinematic_lib common_lib)
target_link_libraries(cpd_kinematic_lib geometry_utils_lib)
target_link_libraries(cpd_kinematic_lib kinematic_base_lib)
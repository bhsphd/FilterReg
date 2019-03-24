add_library(kinematic_base_lib kinematic_model_base.cpp kinematic_model_base.h)
target_link_libraries(kinematic_base_lib common_lib)

add_subdirectory(rigid)
add_subdirectory(affine)
add_subdirectory(cpd)

#The articulated kinematic depends on drake
#add_subdirectory(articulated)

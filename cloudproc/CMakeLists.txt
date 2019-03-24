file(GLOB CLOUD_PROC_SRC *.h *.hpp *.cpp *.cu)
cuda_add_library(cloudproc_lib ${CLOUD_PROC_SRC})
target_link_libraries(cloudproc_lib common_lib)
target_link_libraries(cloudproc_lib geometry_utils_lib)
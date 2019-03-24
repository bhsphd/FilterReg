file(GLOB IMGPROC_SRC *.h *.hpp *.cpp *.cu)
cuda_add_library(imgproc_lib ${IMGPROC_SRC})
target_link_libraries(imgproc_lib common_lib)
#The common utilities
file(GLOB POSER_CORRESPOND_SEARCH_SRC *.h *.hpp *.cpp)
cuda_add_library(corr_common_lib ${POSER_CORRESPOND_SEARCH_SRC})
target_link_libraries(corr_common_lib kinematic_base_lib)
target_link_libraries(corr_common_lib common_lib)
target_link_libraries(corr_common_lib geometry_utils_lib)

#A set of submodules
add_subdirectory(nn_search)
add_subdirectory(gmm)

file(GLOB POSER_NN_SEARCH_SRC *.h *.hpp *.cpp *.cu)

#Exclude the test source
file(GLOB POSER_NN_TEST_SRC *_test.cpp)
exclude(POSER_NN_SEARCH_SRC "${POSER_NN_SEARCH_SRC}" ${POSER_NN_TEST_SRC})

#Build the library
cuda_add_library(nn_search_lib ${POSER_NN_SEARCH_SRC})
target_link_libraries(nn_search_lib common_lib)
target_link_libraries(nn_search_lib corr_common_lib)
file(GLOB POSER_GMM_SRC *.h *.hpp *.cpp *.cu)

#Exclude the test source
file(GLOB POSER_GMM_TEST_SRC *_test.cpp)
exclude(POSER_GMM_SRC "${POSER_GMM_SRC}" ${POSER_GMM_TEST_SRC})

#Build the library
cuda_add_library(gmm_corr_lib ${POSER_GMM_SRC})
target_link_libraries(gmm_corr_lib common_lib)

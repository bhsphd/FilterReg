file(GLOB POSER_COMMON_SRC *.h *.hpp *.cuh *.cpp *.cu)

#Exclude the test source
file(GLOB POSER_COMMON_TEST_SRC *_test.cpp)
exclude(POSER_COMMON_SRC "${POSER_COMMON_SRC}" ${POSER_COMMON_TEST_SRC})

cuda_add_library(common_lib ${POSER_COMMON_SRC})
target_link_libraries(common_lib ${OpenCV_LIBRARIES})
target_link_libraries(common_lib ${PCL_COMMON_LIBRARIES})
target_link_libraries(common_lib ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries(common_lib ${GLOG_LIBRARIES})

#Add test of source if required
if(POSER_BUILD_TEST)
    foreach(test_src ${POSER_COMMON_TEST_SRC})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} "${test_src}")
        target_link_libraries(${test_name} common_lib)
        target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
        if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
            target_compile_features(${test_name} PRIVATE cxx_range_for)
        endif()
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
        #gtest_add_tests(TARGET ${test_name})
    endforeach()
endif()
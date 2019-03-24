file(GLOB POSER_GEOMETRY_SRC *.h *.hpp *.cpp *.cu)

#The test file
file(GLOB POSER_GEOMETRY_TEST_SRC *_test.cpp)
exclude(POSER_GEOMETRY_SRC "${POSER_GEOMETRY_SRC}" ${POSER_GEOMETRY_TEST_SRC})

cuda_add_library(geometry_utils_lib ${POSER_GEOMETRY_SRC})
target_link_libraries(geometry_utils_lib common_lib)

#Add test of source
if(POSER_BUILD_TEST)
    foreach(test_src ${POSER_GEOMETRY_TEST_SRC})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} "${test_src}")
        target_link_libraries(${test_name} common_lib)
        target_link_libraries(${test_name} geometry_utils_lib)
        target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
        if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
            target_compile_features(${test_name} PRIVATE cxx_range_for)
        endif()
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
        #gtest_add_tests(TARGET ${test_name})
    endforeach()
endif()
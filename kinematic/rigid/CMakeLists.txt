file(GLOB RIGID_KINEMATIC_SRC *.h *.hpp *.cpp *.cu)

#The test file
file(GLOB RIGID_TEST_SRC *_test.cpp)
exclude(RIGID_KINEMATIC_SRC "${RIGID_KINEMATIC_SRC}" ${RIGID_TEST_SRC})

#Add the library
cuda_add_library(rigid_kinematic_lib ${RIGID_KINEMATIC_SRC})
target_link_libraries(rigid_kinematic_lib common_lib)
target_link_libraries(rigid_kinematic_lib geometry_utils_lib)
target_link_libraries(rigid_kinematic_lib kinematic_base_lib)


#Add test of source
if(POSER_BUILD_TEST)
    foreach(test_src ${RIGID_TEST_SRC})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} "${test_src}")
        target_link_libraries(${test_name} common_lib)
        target_link_libraries(${test_name} rigid_kinematic_lib)
        target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
        if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
            target_compile_features(${test_name} PRIVATE cxx_range_for)
        endif()
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
        #gtest_add_tests(TARGET ${test_name})
    endforeach()
endif()
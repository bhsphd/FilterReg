# The drake dependency is only required here
set(drake_DIR "/home/wei/Programs/built_src/drake/lib/cmake/drake")
find_package(drake CONFIG REQUIRED)


file(GLOB ARTICULATED_KINEMATIC_SRC *.h *.hpp *.cpp *.cu)
# Exclude the test file
file(GLOB ARTICULATED_KINEMATIC_TEST_SRC *_test.cpp)
exclude(ARTICULATED_KINEMATIC_SRC "${ARTICULATED_KINEMATIC_SRC}" ${ARTICULATED_KINEMATIC_TEST_SRC})

cuda_add_library(articulated_kinematic_lib ${ARTICULATED_KINEMATIC_SRC})
target_link_libraries(articulated_kinematic_lib common_lib)
target_link_libraries(articulated_kinematic_lib drake::drake)
target_link_libraries(articulated_kinematic_lib kinematic_base_lib)
target_link_libraries(articulated_kinematic_lib geometry_utils_lib)
target_link_libraries(articulated_kinematic_lib ${PCL_LIBRARIES})

#Add test of source
foreach(test_src ${ARTICULATED_KINEMATIC_TEST_SRC})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} common_lib)
    target_link_libraries(${test_name} articulated_kinematic_lib)
    target_link_libraries(${test_name} ${GTEST_BOTH_LIBRARIES})
    if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
        target_compile_features(${test_name} PRIVATE cxx_range_for)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    #gtest_add_tests(TARGET ${test_name})
endforeach()
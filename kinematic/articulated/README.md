# Articulated Kinematic

This subdirectory contains the twist-based articulated kinematic model described in our paper. However, the code is developed with a old version of [drake](https://drake.mit.edu/)  for kinematic computation. It is not trivial to build that version of drake as the dependency of drake conflicts with this repository (mainly the `vtk` inside `pcl`). Thus, this subdirectory is not built by default. If you want to build the articulated kinematic model, uncomment the `add_subdirectory` line in `kinematic/CMakeLists.txt`. 

The author is trying to re-factor the code is this subdirectory to work with "modern" drake. Please stay tuned for the update. 
### The CPD kinematic model

The package contains an implementation of the kinematic model in the Coherent Point Drift paper. This kinematic model is much more expensive to compute. Thus, it is better to use this model on (severely) sub-sampled point cloud.

Currently only point-to-point distance is supported, as it is not easy to extract the rotation from the displacement field. 

The planned usage for this package includes online-inference and offline generation. In online case, the observation might be partial, thus PCA based approach might be used. In offline case, the observation should be clean and complete (usually CAD template).
# ROMP - Monocular, One-stage, Regression of Multiple 3D People

ROMP is a one-stage 3D pose estimator of humans. The working principle can be summarized as follows: For each detected human in a scene, the network predicts:
- Parameterized Mesh to describe the joint-angles and limb expansion (in SMPL format)
- Pose of human in current image frame (i.e. how a mesh-rendering has to be scaled, rotated or translated to oberlap the detected human in the scene)

This principle is explained in great detail in their original publication [^1]. In [^2], we give a more compact explanation and we compare ROMP against other state-of-the-art state estimators for our specific AV application. Our project implementation is based on their original GitHub repository [^3]. Their original ReadMe is [here](README_orig.md).

## Contribution

As proposed in [^2], our main contribution is the replacement of their backbone with a smaller architecture in order to reduce the computational load of the AV embedded computer. This means that we modify the deeper layers of the network. Our approach is motivated by the fact that the backbones implemented by [^1] carry ~60 times more parameters than the Head (as shown in the table at the end of this section). More recent pose estimators analyzed in [^2] - especially [^4] - experience good performance by using the EfficientNet architecture. Since this structure promises similar performance using less parameters, we use this architecture to reduce the compute load. 

![contrib_overview](docs/contribution/overview.png)

The specific network architecture we chose is the most recent EfficientNetV2-S which is conveniently available in `pytorch`. For the direct implementation, we took their ResNet-50 as a blueprint (bottom figure left). Since the backbone is the feature extractor in the ROMP application, we do not require the classifier layers. We copy their deconvolution layer-structure to generate the same output layers as before. As the EfficientNet's output-shape differs slightly from the ResNet's, we apply slightly different deconvolution shapes in our contribution. This is summarized here:
![contrib_detail](docs/contribution/resnet_effnet.png)

This implementation cuts the number of parameters in the backbone by 18% and 28% compared to the HRNet-50 and ResNet-50 respectively. We implemented our contribution in order to fit in the original framework such that training, evaluation, and dataset handling can be overtaken directly.

| Component  | # of Parameters  |
|---|---|
| HRNet-32 backbone (in [^1])  | 28,535,552  |
| ResNet-50 backbone (in [^1])  | 32,552,896  |
| EfficientNetV2-S backbone (our contribution)  |  23,496,592 |
| Head  |  568,018 |

Due to the complexity and bugs of their code, we did not implement our second idea of testing an additional loss term.

## Dataset

All relevant datasets as well as the annotations were made available by the authors of [^1] via a [Google-drive folder](https://drive.google.com/drive/folders/1_g4AbXumhufs7YPdTAK3kFMnTQJYs3w3). They also include instructions on the directory structure such that the files can be processed by their code framework [^3] without any problem. See [this page](docs/dataset.md) for the specific procedure. Depending on the dataset location, one must adapt `dataset_rootdir` in the [config.py](romp/lib/config.py).

| Dataset      | Examples Number | Keypoints |
|--------------|-----------------|-----------|
| MuCo         | 677k            | 28        |
| MPII         | 27k             | 16        |
| MPI-INF-3DHP | 627k            | 28        |
| LSP          | 2k              | 14        |
| Human3.6M    | 3.6M            | 32(17)    |
| CrowdPose    | 20k             | 14        |
| COCO         | 47k             | 17        |
| CMU_Panoptic | 1.5M            | 19        |
| AGORA        | 173k            | 66        |
| 3DPW         | 60k             | 17        |
| PoseTrack    | 66k             | 15        |

# Experimental Setup


 
[^1]: Sun, Y., Bao, Q., Liu, W., Fu, Y., Black, M. J., & Mei, T. (2020). Monocular, One-stage, Regression of Multiple 3D People. arXiv preprint arXiv:2008.12272.
[^2]: Milestone 1 Report: https://drive.google.com/file/d/15AhJr35AdtqHdkhOHylhIPvTnorl-QHf/view?usp=drive_link
[^3]: Original GitHub Repository https://github.com/Arthur151/ROMP
[^4]: Sárándi, I., Hermans, A., & Leibe, B. (2022). Learning 3D human pose estimation from dozens of datasets using a geometry-aware autoencoder to bridge between skeleton formats. arXiv preprint arXiv:2212.14474.

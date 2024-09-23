# Less Is More: Label Recommendation for Weakly Supervised Point Cloud Semantic Segmentation 

Zhiyi Pan, Nan Zhang, Wei Gao, Shan Liu, Ge Li

#### Installation Guide

##### Prerequisites

Before installing our program, you should install [PointNeXt](https://github.com/guochengqian/PointNeXt).

##### Installation steps

1. Move all the files of this project to the root directory of PointNeXt.
2. Replace the corresponding files according to the guide.

#### Usages

**Dataset**  The presampling is just to collect all point clouds, area by area and room by room, following [PointNeXt](https://github.com/guochengqian/PointNeXt). You can download our preprocessed ScanNet dataset as follows:

```shell
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1uWlRPLXocqVbJxPvA2vcdQINaZzXf1z_
tar -xvf ScanNet.tar
```

**Inductive Bias Learning and Recommendation**  Please modify the corresponding configuration files to use your own file path. 

For example, train `PointNeXt++` with point cloud upsampling as pretext task, and then recommendation with single-scene clustering strategy (kmeans, the code is based on [ContrastiveSceneContexts](https://github.com/facebookresearch/ContrastiveSceneContexts)) as follows

```shell
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis_LiM/pointnet++_upsampling.yaml visualize=True
```

and using cross-scene clustering strategy as follows

```shell
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py --cfg cfgs/s3dis_LiM/pointnet++_upsampling_batch.yaml visualize=True
```

**Point Cloud Semantic Segmentation Learning** 

```shell
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py cfgs/s3dis/<YOUR_CONFIG> wandb.use_wandb=False mode=test --pretrained_path <YOUR_CHECKPOINT_PATH>
```


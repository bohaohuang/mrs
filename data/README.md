# Data Module
## Features
1. [data_loader](data_loader.py): where pytorch dataset is defined for this framework
2. [data_utils](data_utils.py): some useful functions for creating & managing datasets

## Supported Datasets
It is recommended to use the mean and variance calculated for each dataset in the table below to for the
[configuration file](../config.json). An alternative way is to use the mean and std of the ImageNet dataset.

The function (_get_ds_stats()_) to calculate mean and std of given list of images is defined in [data_utils](data_utils.py).
Either use dataset specific mean&std or not, it is important to have **the same** set of mean&std in both training and evaluation.

| Dataset Name | Label        | Web Page      | Mean | Std |
|:------------:|:------------:|:-------------:|:----:|:---:|
| [Inria](./inria) | Building | [Link](https://project.inria.fr/aerialimagelabeling/) |(0.41776216, 0.43993309, 0.39138562) | (0.18476704, 0.16793099, 0.15915148) |
| [DeepGlobeBuilding](./deepglobe)| Building | [Link](https://competitions.codalab.org/competitions/18544) | (0.34391829, 0.41294382, 0.45617783) | (0.10493991, 0.09446405, 0.08782307) |
| [DeepGlobeRoad](./deepgloberoad)| Road | [Link](https://competitions.codalab.org/competitions/18467) | (0.40994515, 0.38314009, 0.28864455) | (0.12889884, 0.10563929, 0.09726452) |
| [DeepGlobeLand](./deepglobeland) | Urban land, Agriculture land, Rangeland, Forest land, Water, Barren land | [Link](https://competitions.codalab.org/competitions/18468) | (0.38475783 0.34792641 0.24514727) | (0.12838193 0.10063904 0.08192587)
| [MIT Road](./mnih) | Road | [Link](https://www.cs.toronto.edu/~vmnih/data/) | (0.4251811, 0.42812928, 0.39143909) | (0.22423858, 0.21664895, 0.22102307) |
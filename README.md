# Uncertainty estimation for deep supervised models

My master thesis project.

[Project plan (in Croatian)](https://docs.google.com/document/d/1XF4h3DU0lrqXeNxkvuxOQtaGkMY9FSaiX9uu45-i58g)

### Related (uncertainty, bayesian neural networks, out-of-distribution and misclassified examples)
  * MacKay
  * [neal95](https://pdfs.semanticscholar.org/db86/9fa192a3222ae4f2d766674a378e47013b1b.pdf) Bayesian deep learning for neural networks
  * [gal15arxiv](https://arxiv.org/abs/1506.02142) Dropout as a Bayesian Approximation
  * [gal16cam](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf) Uncertainty in Deep Learning
  * [kendall17arxiv](https://arxiv.org/abs/1703.04977) What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision
  * [smith18arxiv](http://arxiv.org/abs/1803.08533) What Understanding Measures of Uncertainty for Adversarial Example Detection
  * [hendryks2017arxiv](https://arxiv.org/abs/1610.02136) A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
  * [guo17arxiv](https://arxiv.org/abs/1706.04599) On Calibration of Modern Neural Networks
  * [liang18iclr](https://openreview.net/forum?id=H1VGkIxRZ&noteId=r1OWfeiXf) Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks
  * [devries18arxiv](https://arxiv.org/abs/1802.04865) Learning Confidence for Out-of-Distribution Detection in Neural Networks
  * [lee18iclr](https://openreview.net/forum?id=ryiAv2xAZ) Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples

### Code structure
```
.
├── data  # non-code
│   ├── nets                # trained model parameters
│   └── plots
├── dl_uncertainty          # library
│   ├── data
│   │   ├── data.py         # `Dataset`, `DataLoader`
│   │   ├── datasets.py     # raw `Dataset` implementations
│   ├── ioutils
│   ├── models
│   │   ├── components.py   # abstractions/wrappers of `tf_utils/layers.py` with an experimantal programming design
│   │   ├── modeldef.py     # `InferenceComponent` (model graph), `TrainingComponent` (inductive bias), `ModelDef` ((almost) all information that needs to be given to the `Model` inintializer)
│   │   ├── model.py        # `Model` with methods such as `train`, `test`, `predict`, `save_state`, `load_state`, `load_parameters`
│   │   └── tf_utils
│   │       ├── evaluation.py
│   │       ├── layers.py   # functional core implementtions of layers and larger model components
│   │       ├── losses.py
│   │       ├── regularization.py
│   │       ├── update_ops.py
│   │       └── variables.py
│   ├── processing          # data augmentation, preprocessing, ...
│   ├── utils               # visualization, ...
│   ├── data_utils.py       # dataset factories `get_dataset` and `get_cached_dataset_with_normalized_inputs`
│   ├── dirs.py  
│   ├── evaluation.py
│   ├── model_utils.py      # factories `get_inference_component`, `get_training_component`, model factory `get_model`
│   ├── parameter_loading.py
│   └── training.py         # `train`, training loop function
└── scripts
    ├── _context.py
    ├── test.py             # e.g. python test.py cifar wrn 28 10 ../data/nets/cifar-trainval/wrn-28-10-e100/2018-05-28-0956/Model
    ├── train.py            # e.g. python train.py cifar rn 18 64 --epochs 100 --trainval
    └── view_dataset.py     # e.g. python view_dataset.py cifar trainval
```

### Prerequisites

`dl_uncertainty/dirs.py` requires that directories matching regular expressions below exist.
```
.(\/..)+\/data\/cache       # some ancestor directory contains `data/cache`
.(\/..)+\/data\/datasets    # some ancestor directory contains `data/datasets` ...
.(\/..)+\/data\/pretrained_parameters
.(\/..)+\/data\/nets
```

#### Dataset directory

The `<ancestor>/data/datasets` directory (stored in the `dirs.DATASETS` variable in `dl_uncertainty/dirs.py`) needs to contain the required datasets. For example, for CIFAR-10 it needs to contain this directory subtree:
```
cifar-10-batches-py/
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch
```

#### Preprocessed data caching directory

The `<ancestor>/data/cache` directory is used for caching preprocessed data in the `get_cached_dataset_with_normalized_inputs` factory function  in `dl_uncertainty/data_utils.py` (`data_utils.LazyNormalizer` stores average image channel values and `data.Dataset.cache_hdd_only` stores preprocessed data). `python scripts/clear_cache.py <dataset_id>` can be used to clear cache for a dataset.

#### Trained model parameters directory

The `<ancestor>/data/nets` directory is used storing trained model parameters and training log data.


### Usage

#### Training

To train a model, the script `scripts/train.py` needs to be run. An example command for training a model is 
```
python train.py <dataset_id> <model_id> <model_depth> <model_width_parameter> --epochs <epoch_count> [--trainval]
```

Some examples of `<dataset_id>` are `cifar`, `inaturalist`, `tinyimagenet`. Other dataset IDs can be seen in `dl_uncertainty/data_utils.py`.

`<model_id> <model_depth> <model_width_parameter>` describe the general model structure. Some examples are 
  * ResNets (V2), where `<model_width_parameter>` represents the number of feature maps of the first group of residual blocks  `rn 18 64` (ResNetV2 18), `rn 50 64` (ResNetV2 50);
  * wide ResNets, where `<model_width_parameter>` represents 1/16 of the number of feature maps of the first group of residual blocks: `wrn 28 10` (WRN-28-10), `wrn 16 4` (WRN-16-4);
  * DenseNets (BC variant) where `<model_width_parameter>` represents the growth rate (k): `dn 100 12`, `dn 121 32`.

If the optional `--trainval` argument is passed, the model is trained on the `trainval` set and tested on the `test` set. Otherwise, it is trained on the `train` set and tested on the `val` set.

Even though `--epochs` looks like it is optional, it is not.

When the training is finished the path of the trained model (a path ending with "/Model") is saved is printed on the screen. It can be copied for use in other scripts, like `test.py`.

#### Testing

To test a trained model on some new dataset `scripts/test.py` needs to be run. An example command for testing is 
```
python test.py <dataset_id> <model_id> <model_depth> <model_width_parameter> <path_to_model>
```

`<dataset_id>` is the ID of the dataset that the model was trained on.
`<path_to_model>` is the path to the saved parameters of the model (the path ending with "/Model)).

TODO


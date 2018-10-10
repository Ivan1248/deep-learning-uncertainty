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

### Usage
TODO


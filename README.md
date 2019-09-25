# Models for Remote Sensing
## TODOs
- [ ] Encoder Structures:
    - [X] VGG
    - [X] ResNet
    - [ ] DenseNet
    - [ ] SqueezeNet
    - [ ] InceptionNet
- [ ] Decoder Structures:
    - [X] UNet
    - [ ] LinkNet (https://github.com/e-lab/pytorch-linknet)
    - [X] PSPNet
    - [ ] DeepLabV3
- [ ] Different Losses:
    - [X] Xent
    - [ ] Jaccard Approximation
    - [ ] Lovasz softmax (https://github.com/bermanmaxim/LovaszSoftmax/tree/master/pytorch)
    - [ ] Weighted combination of arbitrary supported losses
- [X] Multi GPU Training
- [ ] Evaluation
    - [X] Dataset Evaluator
    - [ ] Evaluate Report & Prediction Map
- [X] Toy Dataset
- [ ] Config as json file
- [ ] Check flex loading function
## Known Bugs
- [ ] Unable to do model-wise data parallel
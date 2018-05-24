# PytorchCNN
Pytorch CNN Benchmark, From *Flash Lab, Gatech*

## Methodology

### Regularization
**Orth Reg.**: Add  Orthogonal Regularization  
**Sphere**: Sphere Net  
**Spectral Normalization (SN)**:  
**D-optimal Reg**:  
**Incoherence Reg**



## Results

### CIFAR10 Benchmark

| Model                     | Plane | Orth  | Sphere| SN    | D-opt | Incoh |
| -------------------       | ----- | ----- | ----- | ----- | ----- | ----- |
| ResNet-20                 | 92.55 | 91.61 |       | 92.37 |
| ResNet-32                 | 93.01 |
| WResNet-20                | 95.29 | 94.89 |       | 95.31 |
| WResNet-32                | 95.43 |

### CIFAR100 Benchmark

| Model                     | Plane | Orth  | Sphere| SN    | D-opt | Incoh |
| -------------------       | ----- | ----- | ----- | ----- | ----- | ----- |
| ResNet-20                 | 68.51 | 67.88 |       | 68.56 |
| ResNet-32                 | 70.84 |
| WResNet-20                | 77.71 | 75.23 |       | 77.29 |
| WResNet-32                | 79.04 |

### ImageNet Benchmark

## Architectures

- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [Pre-act-ResNet](https://arxiv.org/abs/1603.05027)
- [x] [ResNeXt](https://arxiv.org/abs/1611.05431) (Imported from [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch))
- [x] [Wide Residual Networks](http://arxiv.org/abs/1605.07146) (Imported from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch))
- [x] [DenseNet](https://arxiv.org/abs/1608.06993)

## TODO List

## Acknowledge

Some code comes from https://github.com/bearpaw/pytorch-classification

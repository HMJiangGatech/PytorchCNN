# PytorchCNN
Pytorch CNN Benchmark, From *Flash Lab, Gatech*

## Methodology

### Regularization
**Orth Reg.**: Add  Orthogonal Regularization  
**Sphere**: Sphere Net  
**Spectral Normalization (SN)**:  
**D-optimal Reg**:  



## Results

### CIFAR10 Benchmark

| Model                     | Plane | Orth  | Sphere| SN    | D-opt |
| -------------------       | ----- | ----- | ----- | ----- | ----- |
| alexnet                   | 22.78 |
| vgg19_bn                  | 6.66  |
| ResNet-110                | 6.11  |
| PreResNet-110             | 4.94  |
| WRN-28-10 (drop 0.3)      | 3.79  |
| ResNeXt-29, 8x64          | 3.69  |
| ResNeXt-29, 16x64         | 3.53  |
| DenseNet-BC (L=100, k=12) | 4.54  |
| DenseNet-BC (L=190, k=40) | 3.32  |

### CIFAR100 Benchmark

| Model                     | Plane | Orth  | Sphere| SN    | D-opt |
| -------------------       | ----- | ----- | ----- | ----- | ----- |
| alexnet                   | 56.13 |
| vgg19_bn                  | 28.05 |
| ResNet-110                | 28.86 |
| PreResNet-110             | 23.65 |
| WRN-28-10 (drop 0.3)      | 18.14 |
| ResNeXt-29, 8x64          | 17.38 |
| ResNeXt-29, 16x64         | 17.30 |
| DenseNet-BC (L=100, k=12) | 22.88 |
| DenseNet-BC (L=190, k=40) | 17.17 |

### ImageNet Benchmark

## Architectures

- [x] [AlexNet](https://arxiv.org/abs/1404.5997)
- [x] [VGG](https://arxiv.org/abs/1409.1556) (Imported from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar))
- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [Pre-act-ResNet](https://arxiv.org/abs/1603.05027)
- [x] [ResNeXt](https://arxiv.org/abs/1611.05431) (Imported from [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch))
- [x] [Wide Residual Networks](http://arxiv.org/abs/1605.07146) (Imported from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch))
- [x] [DenseNet](https://arxiv.org/abs/1608.06993)

## TODO List

## Acknowledge

Some code comes from https://github.com/bearpaw/pytorch-classification

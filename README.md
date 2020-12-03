# Chair Segments
Chair Segments: A Compact Benchmark for the Study of Object Segmentation

### Abstract
Over the years, datasets and benchmarks have had an outsized influence on the design of novel algorithms. In this paper, we introduce **ChairSegments**, a novel and compact semi-synthetic dataset for object segmentation. We also show empirical findings in transfer learning that mirror recent findings for image classification. We particularly show that models that are fine-tuned from a pretrained set of weights lie in the same basin of the optimization landscape. ChairSegments consists of a diverse set of prototypical images of chairs with transparent backgrounds composited into a diverse array of backgrounds. We aim for ChairSegments to be the equivalent of the CIFAR-10 dataset but for quickly designing and iterating over novel model architectures for segmentation. On Chair Segments, a U-Net model can be trained to full convergence in only thirty minutes using a single GPU. Finally, while this dataset is semi-synthetic, it can be a useful proxy for real data, leading to state-of-the-art accuracy on the Object Discovery dataset when used as a source of pretraining.

### Requirements
- Python 3
- Pytorch > 1.0
- torchVision

### Download ChairSegment dataset
```
sh download_data.sh
```

### Training
```
# Start training with: 

# ChairSegment -  parameters

python main.py --lr=1e-3 --arch=unet --optimizer=Adam --epochs=20
python main.py --lr=1e-4 --arch=fcnvgg16 --optimizer=SGD --epochs=50
python main.py --lr=1e-4 --arch=fcnresnet50 --optimizer=SGD --epochs=100 --momentum=0.9 --weight_decay=1e-5
python main.py --lr=1e-6 --arch=fcnresnet101 --optimizer=RMSprop --epochs=100 --momentum=0.9 --weight_decay=1e-7
```

### Chair Segments
| Model             | Prec.       |IoU.        |Dice        |
| ----------------- | ----------- |----------- |----------- |
| Unet                 | 97.18%      | 85.08%      | 91.25%      |
| FCN-VGG-16           | 91.73%      | 61.09%      | 74.09%      |
| FCN-ResNet-50        | 92.04%      | 60.19%      | 72.58%      |
| FCN-ResNet-101       | 92.19%      | 61.62%      | 73.96%      |


Report: [https://arxiv.org/abs/2012.01250](https://arxiv.org/abs/2012.01250)

```
@article{yang-etal-2020-using,
  title = {Chair Segments: A Compact Benchmark for the Study of Object Segmentation},
  author = {Leticia Pinto-Alva, Ian K. Torres, Rosangel Garcia, Ziyan Yang, Vicente Ordonez},
  year = {2020}
  address = {Online}
  url = {https://arxiv.org/abs/2012.01250}
}
```

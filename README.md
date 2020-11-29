# Chair Segments
Chair Segments: A Compact Benchmark for the Study of Object Segmentation

### Abstract
Over the years, datasets and benchmarks have had an outsized influence on the design of novel algorithms. In this paper, we introduce **ChairSegments**, a novel and compact semi-synthetic dataset for object segmentation. We also show empirical findings in transfer learning that mirror recent findings for image classification. We particularly show that models that are fine-tuned from a pretrained set of weights lie in the same basin of the optimization landscape. ChairSegments consists of a diverse set of prototypical images of chairs with transparent backgrounds composited into a diverse array of backgrounds. We aim for ChairSegments to be the equivalent of the CIFAR-10 dataset but for quickly designing and iterating over novel model architectures for segmentation. On Chair Segments, a U-Net model can be trained to full convergence in only thirty minutes using a single GPU. Finally, while this dataset is semi-synthetic, it can be a useful proxy for real data, leading to state-of-the-art accuracy on the Object Discovery dataset when used as a source of pretraining.

### Requirements
- Python 3
- Pytorch > 1.0

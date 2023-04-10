# SSFormer
##  This is an official implementation of "Learning the degradation distribution for medical image superresolution via sparse swin transformer".

<p align='center'>  
  <img src='fig/1.png' width='440'/>
</p>



# Introduction

Clear medical images are significant for auxiliary diagnoses, but the images generated by various medical devices inevitably contain considerable noise. Although various models have been proposed for denoising, these methods ignore that different types of medical images have different noise levels, which leads to unsatisfactory test results. In addition, collecting a large number of medical images for training denoising models consumes many material resources. To address these issues, we formulated a progressive denoising architecture that contains preliminary and profound denoising. First, we construct a noise level estimation network to estimate the noise level via self-supervised learning and perform preliminary denoising with a dilated blind-spot network. Second, with the learned noise distribution, we synthesize noisy natural images to construct clean-noisy natural image pairs. Finally, we design a novel medical image denoising model for profound denoising by training these pairs. The proposed three-stage learning scheme and progressive denoising architecture not only solve the problem that the denoising model only adapts to a single noise level but also alleviate the lack of medical image pairs. Moreover, we integrate dense attention and sparse attention to constitute the retractable transformer module into the profound denoising model, which reconciles a wider receptive field and enhances the representation ability of the transformer. This allows the denoising model to obtain retractable attention on the input feature and can capture more local and global receptive fields simultaneously. The results of qualitative and quantitative experiments demonstrate that our method outperforms other denoising methods in terms of both qualitative and quantitative aspects.

## Quick Test
#### Dependencies
- Python 3
- [PyTorch >= 0.4.0](https://pytorch.org/)


### Test models
1. Clone this github repo. 
```
git clone https://github.com/codehxj/SSFormer/
cd SSFormer
```
2. Place your own **low-resolution images** in `./LR` folder.  
3. Download pretrained models from [Baidu Drive] (Later...). Place the models in `./models`. 
4. Run test. We provide SSFormer model at scale "x 2" and "x 4".
```
python test.py models/SSFormer_x2.pth
python test.py models/SSFormer_x4.pth
```
5. The results are in `./results` folder.


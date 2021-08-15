# Understanding-Transformer-Self-Attention-Maps

This repo is a deeper dive into Transformer Encoders' Self Attention maps, what they are and why are they are capable of some form of instance segmentation.  

The code is adapted from Facebook's Detection Transformer (DETR), specifically the tutorial, [detr_hands_on (https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb "DETR Colab"). 

## Background on DETR
![image](https://user-images.githubusercontent.com/79006977/129480190-77d1d767-85f0-4913-b6e0-6a99941eb2ad.png)
DETR passes the image through a backbone CNN to obtain the representation of it.  


![image](https://user-images.githubusercontent.com/79006977/129473955-cfcf9b4a-8748-42c4-91b9-a812d1c68f6d.png)
Quote from Facebook AI Research Engineer @fmassa:  ...'In the end, it's a matter of deciding if you want the attention of all pixels at location x, or the contribution of location x to all the attention maps.'
 Issue: https://github.com/facebookresearch/detr/issues/162

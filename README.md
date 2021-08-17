# Understanding-Transformer-Self-Attention-Maps

The code is adapted from Facebook's Detection Transformer (DETR), specifically the tutorial, [detr_hands_on (https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb "DETR Colab"). 

The DETR paper and others have demonstrated that the self attention weights/maps are capable of some form of instance segmentation. This is an attempt to illustrate how self attention works, and why it is able to achieve this. 

----> INSTANCE SEG EXAMPLE

## Background on DETR
![image](https://user-images.githubusercontent.com/79006977/129480190-77d1d767-85f0-4913-b6e0-6a99941eb2ad.png)

DETR passes the input image through a backbone CNN to obtain the representation of it. The model used in the Facebook tutorial adopted the Resnet50 as a backbone, downsampling the H and W of the image by 32x. A pointwise Conv2D was then used to change the number of input channels i.e. 3, to the embedding dimension of 256, with the final transformer input token shape of (HxW/32^2, batch_size, 256). 

Unlike the Vision Transformer, DETR does not use a linear layer to project the image tokens into individual q,k,v feature spaces. Rather, the query, key, value matrices inputs to the transformer are identical, except that q and k are concatenated with a positional embedding. Interestingly, this allows for DETR to be partially 'image size/aspect ratio invariant', unlike the Vision Transformer, where its input images require a centre crop.

## Self Attention Weights/Maps
It is useful to understand that the Self Attention Maps are simply derived from a matrix multiplication of q and k Transpose. Each q row vector and k.T column vector represents each pixel of the backbone output feature map. The q columns and k.T rows are the corresponding embeddings of each pixel. It is intuitive to understand these embeddings as features of that pixel, as well as to borrow from NLP what do the features mean. 

![image](https://user-images.githubusercontent.com/79006977/129739480-53d8f810-a617-4bd2-82c3-b9f63505541b.png)

(Image taken from [Rasa's YouTube Video, Transformers & Attention 1: Self Attention](https://www.google.com)

In NLP, each element of the word embedding can be intuited to be 'categories' e.g. family, royalty, power etc. The more related the word is to a category, the higher the corresponding value of that vector element that represents that category. Hence, similarity between two words can simply be obtained by getting the dot product of the first word's embedding vector with the second word's transposed embedding.  Dot product multiplies the elements of the two words belonging to the same category, and sums up all these similarity values across the two words. If both words are closely related, they would have large values in the same elements of their embedding vectors, and the dot product of their embedding vectors will result in a high magnitude.

For vision, perhaps the emebedding dimension can be interpreted as different 'characteristics' of the pixel e.g. its color, position etc. Hence the similarity between two pixels can be derived from its dot product. 

![Self Attn Maps](https://user-images.githubusercontent.com/79006977/129746777-fd4347b6-0f30-411e-a989-24ac02b60481.png)

The Self Attention Map is simply derived from the dot product of each Query vector (representing a pixel) with every other Key vector (also representing a pixel), and the resulting scalar value is how closely related the 2 pixels are. The first row in the Self Attention Map represents the 'queried similarity' values between the first pixel of the feature map, and every other pixel. It can be said that the first row represents how much attention the first pixel of the feature map pays to every other pixel. The columns of the map represents how much attention every pixel pays to the first pixel. 

## 

![image](https://user-images.githubusercontent.com/79006977/129473955-cfcf9b4a-8748-42c4-91b9-a812d1c68f6d.png)
Quote from Facebook AI Research Engineer @fmassa:  ...'In the end, it's a matter of deciding if you want the attention of all pixels at location x, or the contribution of location x to all the attention maps.'
 Issue: https://github.com/facebookresearch/detr/issues/162

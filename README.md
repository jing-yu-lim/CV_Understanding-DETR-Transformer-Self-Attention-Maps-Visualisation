# Understanding-DETR-Transformer-Self-Attention-Maps

The code is adapted from Facebook's Detection Transformer (DETR), specifically the tutorial, [detr_hands_on] (https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb "DETR Colab"). 

The DETR paper and others have demonstrated that the self attention weights/maps are capable of some form of instance segmentation. This is an attempt to illustrate how self attention works, and why it is able to achieve this. 



## Background on DETR Encoder

<p align="center">
<img width="300" height="200" src="https://user-images.githubusercontent.com/79006977/129480190-77d1d767-85f0-4913-b6e0-6a99941eb2ad.png">
</p>


DETR passes the input image through a backbone CNN to obtain the representation of it. The model used in the Facebook Colab tutorial adopted the Resnet50 as a backbone, which downsamples the H and W of the image by 32x. A pointwise Conv2D is then used to change the number of input channels i.e. 3, to the embedding dimension i.e. 256. The H and W are then flattened, and final transformer input has the shape of (H/32 * W/32, 256), which forms the Query(Q), Key(K), Value(V) matrices of the DETR encoder. 

Unlike the Vision Transformer, DETR does not use a linear layer to project the image tokens into individual Q, K, V feature spaces. Rather, Q, K, V matrices inputs to the transformer are identical, except that q and k are concatenated with a positional embedding. Interestingly, this allows for DETR to be partially 'image size/aspect ratio invariant', unlike the Vision Transformer, where its input images require a centre crop.

## Self Attention Weights
It is useful to understand that Self Attention Weights are simply derived from a matrix multiplication between Q (H/32 * W/32, 256) and K transposed (256, H/32 * W/32). Each Q row vector and K.T column vector represents each pixel of the backbone output feature map. Each Q columns and K.T rows are the corresponding embeddings of each pixel. An intuitive understanding of the  pixels' embedding can be borrowed from NLP. 

In NLP, each element of the word embedding vector can be intuited to be 'categories' e.g. family, royalty, power etc (shown below). The more related the word is to a category, the higher the value of that vector element corresponding to that category. Hence, similarity between two words can simply be obtained by getting the dot product of the first word's embedding vector with the second word's transposed embedding vector.  Dot product multiplies the elements in the same position in the embedding vectors i.e same 'category' from the two words, and sums up all these similarity values across the two words. If both words are closely related, they would have embedding vectors with large values in the same positions, and the dot product of their embedding vectors will result in a high magnitude.

<p align="center">
<img width="337" height="228" src="https://user-images.githubusercontent.com/79006977/129739480-53d8f810-a617-4bd2-82c3-b9f63505541b.png">
</p>

(Image taken from [Rasa's YouTube Video, Transformers & Attention 1: Self Attention](https://www.youtube.com/watch?v=yGTUuEx3GkA&t=489s))

For DETR, perhaps the emebedding dimension can be intuited as different 'characteristics' of the pixel e.g. its color, position etc, and the Conv Backbone and the pointwise Conv layer extracted these 'characteristics' from the raw image. Hence the similarity between two pixels' embedding vector can be derived from its dot product. 

![Self Attn Weights](https://user-images.githubusercontent.com/79006977/129818679-ccdec0e3-a05c-4b64-85d4-9b146f07f396.png)

(please bear with the poor handwriting)

The Self Attention Weight is simply derived from the dot product of each Query vector (representing a pixel) with every other Key vector (also representing a pixel), and the resulting scalar value is representative of how closely related the 2 pixels are. The first row in the Self Attention Weights represents the 'queried similarity values' between the first pixel of the feature map, and every other pixel. It can be said that the first row represents how much attention the first pixel of the feature map pays to every other pixel. The columns of the map represents how much attention every pixel pays to the first pixel. 

## Understanding the Facebook DETR Tutorial on Attn Map Visualisation

In the FB Tutorial, the Self Attn Weights with shape of (H/32 * W/32, H/32 * W/32), was reshaped into (H/32, W/32, H/32, W/32). For convenience, we shall assign:
* sattn_shape1 = Attn Weight with shape (H/32, W/32, H/32, W/32) 
* sattn_shape2 = Attn Weight with shape (H/32 * W/32, H/32 * W/32) 

To obtain a Self Attention Map, which some times does instance segmentation, the FB Tutorial simply sliced sattn_shape2 using [Y/32, X/32, ... ] or [ ... , Y/32, X/32], where (X,Y) is a chosen coordinate on the raw image. e.g. the Self Attention Map of the coordinate 0,0 can derived using sattn_shape1[Y/32, X/32, ...]  OR sattn_shape1[ ... , Y/32, X/32] 

When I first saw this, it seemed like black magic to me. The only explanation I could find was from the [DETR issues forum](https://github.com/facebookresearch/detr/issues/162), where a FB Engineer (@fmassa) commented: ...'In the end, it's a matter of deciding if you want the attention of all pixels at location x, or the contribution of location x to all the attention maps.' And I was thus inspired to dig deeper. 

We can understand why this works by understanding how to map between the two shapes of the Self Attention Weights (H/32, W/32, H/32, W/32) -> (H/32 * W/32 , H/32 * W/32) and building on the understanding of Self Attention Weights from the above section. 

It turns out that given a coordinate (X,Y) the following slices of the 2 differently shaped attention weights yields the same attention map, using the conversion below:   
### Conversion of slicing between sattn_shape1 and sattn_shape2
Meaning | sattn_shape1.shape: <br /> (H/32, W/32, H/32, W/32) | sattn_shape2.shape: <br /> (H/32 * W/32 , H/32 * W/32)
--- | --- | ---
Attn that (X,Y) pays to every other pixel | sattn_shape1[Y/32 , X/32 , ... ] | sattn_shape2[Y/32 * W/32 + X/32 , ... ]
Attn that every other pixel pays to (X,Y) | sattn_shape1[ ..., Y/32 , X/32] | sattn_shape2[ ... , Y/32 x W/32 + X/32]

<p align="center">
<img width="1000" height="480" src="https://user-images.githubusercontent.com/79006977/129822169-5f0b3f53-cf0e-4e1b-bc6d-c3c7e09dcc29.png">
</p>
From @fmassa's comment, it can be interepreted that the first two dimension of sattn_shape1 (H/32, W/32, H/32, W/32 ) as an index to the respective attention map, specifically the attention Coord(X,Y) pays to every other pixel. We know from the previous section that the rows of the sattn_shape2 weights correspond to the attention a pixel pays to every other pixel. Hence to obtain the correct row number of Coord(X,Y), we can flatten the H/32 and W/32 dimension, giving us Y/32 * W/32 + X/32. 

Conversely, the last 2 dimension of the sattn_shape1 is an index to the respective attention map, specifically the attention every other pixel pays to Coord(X,Y), and the row number derived earlier is the column number for sattn_shape2 to derive the respective attention map

<p align="center">
<img width="1100" height="480" src="https://user-images.githubusercontent.com/79006977/129822097-93045ca7-7897-4d43-b4cd-38bb6011044e.png">
</p>



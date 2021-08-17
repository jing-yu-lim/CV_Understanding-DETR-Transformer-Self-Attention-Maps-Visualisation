# Understanding-Transformer-Self-Attention-Maps

The code is adapted from Facebook's Detection Transformer (DETR), specifically the tutorial, [detr_hands_on (https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb "DETR Colab"). 

The DETR paper and others have demonstrated that the self attention weights/maps are capable of some form of instance segmentation. This is an attempt to illustrate how self attention works, and why it is able to achieve this. 

----> INSTANCE SEG EXAMPLE

## Background on DETR Encoder

<p align="center">
<img width="600" height="450" src="https://user-images.githubusercontent.com/79006977/129480190-77d1d767-85f0-4913-b6e0-6a99941eb2ad.png">
</p>


DETR passes the input image through a backbone CNN to obtain the representation of it. The model used in the Facebook Colab tutorial adopted the Resnet50 as a backbone, which downsamples the H and W of the image by 32x. A pointwise Conv2D was then used to change the number of input channels i.e. 3, to the embedding dimension i.e. 256. The H and W are then flattened, and final transformer input has the shape of (H/32 x W/32, 256), which forms the query(q), key(k), value(v) matrices of the DETR encoder. 

Unlike the Vision Transformer, DETR does not use a linear layer to project the image tokens into individual Q, K, V feature spaces. Rather, Q, K, V matrices inputs to the transformer are identical, except that q and k are concatenated with a positional embedding. Interestingly, this allows for DETR to be partially 'image size/aspect ratio invariant', unlike the Vision Transformer, where its input images require a centre crop.

## Self Attention Weights/Maps
It is useful to understand that a Self Attention Map is simply derived from a matrix multiplication between Q (H/32 x W/32, 256) and K transposed (256, H/32 x W/32). Each Q row vector and K.T column vector represents each pixel of the backbone output feature map. Each Q columns and K.T rows are the corresponding embeddings of each pixel. An intuitive understanding of the  pixels' embedding can be borrowed from NLP. 

<p align="center">
<img width="337" height="228" src="https://user-images.githubusercontent.com/79006977/129739480-53d8f810-a617-4bd2-82c3-b9f63505541b.png">
</p>

(Image taken from [Rasa's YouTube Video, Transformers & Attention 1: Self Attention](https://www.youtube.com/watch?v=yGTUuEx3GkA&t=489s))


In NLP, each element of the word embedding vector can be intuited to be 'categories' e.g. family, royalty, power etc. The more related the word is to a category, the higher the value of that vector element that corresponds to that category. Hence, similarity between two words can simply be obtained by getting the dot product of the first word's embedding vector with the second word's transposed embedding vector.  Dot product multiplies the elements in the same position in the embedding vectors i.e same 'category' from the two words, and sums up all these similarity values across the two words. If both words are closely related, they would have embedding vectors with large values in the same positions, and the dot product of their embedding vectors will result in a high magnitude.

For DETR, perhaps the emebedding dimension can be intuited as different 'characteristics' of the pixel e.g. its color, position etc. Hence the similarity between two pixels can be derived from its dot product. 

![Self Attn Maps](https://user-images.githubusercontent.com/79006977/129746777-fd4347b6-0f30-411e-a989-24ac02b60481.png)

The Self Attention Map is simply derived from the dot product of each Query vector (representing a pixel) with every other Key vector (also representing a pixel), and the resulting scalar value is how closely related the 2 pixels are. The first row in the Self Attention Map represents the 'queried similarity values' between the first pixel of the feature map, and every other pixel. It can be said that the first row represents how much attention the first pixel of the feature map pays to every other pixel. The columns of the map represents how much attention every pixel pays to the first pixel. 

## Understanding the Facebook DETR Tutorial on Attn Map Visualisation

In the Colab Tutorial, the Self Attn Weights with shape of (H/32 x W/32, H/32 x W/32), was reshaped into (H/32, W/32, H/32, W/32). Thereafter, a coordinate can be mapped from the original image, by substituting the H and W values into the first 2 dim of the Self Attention Weights, and the result is the corresponding Self Attention Map, which some times does instance segmentation. e.g. To obtain the Self Attention Map of the coordinate 0,0, use index [0, 0, ...] to slice the Self Attention Weights. When I first saw this I was flabbergasted as it seemed like black magic to me. The only explanation I could find was from the [DETR issues forum](https://github.com/facebookresearch/detr/issues/162), where a FB Engineer (@fmassa) commented on why either substituting [H/32 , W/32,...] or [...,H/32, W/32] works: ...'In the end, it's a matter of deciding if you want the attention of all pixels at location x, or the contribution of location x to all the attention maps.'  

And I was thus inspired to make this repo. 

We can understand why this works by understanding how to map between the two shapes of the Self Attention Weights (H/32,W/32,H/32,W/32) -> (H/32 x W/32 , H/32 x W/32) and building on understanding of Self Attention Weights from the above section. 
<p align="center">
<img width="650" height="480" src="https://user-images.githubusercontent.com/79006977/129764041-0e8d0cfe-1c68-4f87-beae-84e9ebc90dc7.png">
</p>

It turns out that given a coordinate (X,Y) the following slices of the 2 differently shaped attention weights yields the same attention map (hopefully the graphic above helps with the intuition) :   

Attn Weight Shape: (H/32,W/32,H/32,W/32) | Attn Weight Shape: (H/32 x W/32 , H/32 x W/32)
--- | ---
[Y, X, ...] | [Y x W/32 + X, ...]
[..., Y,X] | [..., Y x W/32 + X]

![image](https://user-images.githubusercontent.com/79006977/129774867-06ff003c-4838-4cdc-9023-b06589c80b76.png)

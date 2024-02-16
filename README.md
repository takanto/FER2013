# FER2013
This project is designed to provide developers with a range of pretrained models trained on FER2013. The project includes TensorFlow implementations of sophisticated deep learning models, Google Colab Notebooks, and TFLite files. 

## Dataset
FER2013 comprises 35,887 RGB images of faces, each with dimensions (48, 48, 3). These images represent 7 distinct emotions: happy, sad, angry, surprised, disgust, fear, and neutral (Verma, R., 2018). The dataset was curated by conducting a Google image search for each emotion and its synonyms (Quinn, M., Sivesind, G., Reis, G., n.d.).

## Data Preprocessing
The images undergo augmentation through flipping and random cropping. This process aims to expand the dataset and enhance model robustness. In the context of GNNs, facial landmarks are extracted to construct graphs using Dlib facial landmark detection model (shape_predictor_68_face_landmarks.dat). The resulting graphs have 68 nodes containing x, y coordinates as features. See the Google Colab Notebooks stored in notebook folder for implementation details. 

## Models
The following is the list of models available in this repository. 

### CNNs
1. **VGG16**: A highly effective pretrained model for image classification tasks, containing 16 CNN blocks. 
2. **ResNet50**: A highly effective pretrained model for image classification tasks, utilising residual connections. 
3. **MobileNetV2**: A highly effective pretrained model for image classification tasks, made to be suitable for mobile applications. 
4. **Convolutional Block Attention Module**: CNN implementaton of spatial attention and channel attention. 


### Transformers
1. **Vision Transformer**: An implementation of transformer in computer vision task, made possible by creating image patches. The default is to have input size (224, 224, 3), but the implementation allows variable sie of input size and patch size, including (48, 48, 3). 
2. **Hybrid Vision Transformer (MobileNetV2)**: An implementation of transformer in computer vision task, made possible by leveraging a pretrained CNN model for dimentionality reduction. 
3. **Swin Transformer**: A transformer model that utlises W-MSA and SW-MSA for efficient self-attention. 
4. **Hybrid Attention Transformer**: A variant of Swin transformer that adds channel attention and overlapping cross attention for a larger receptive field. The channel attention is modified to channel attention from ECANet. 

### Graph Neural Networks
1. **Graph Convolutional Network**: Known as a state-of-the-art model for graph data structures, Graph Convolutional Network (GCN) excels in handling graph-based data.
2. **Graph Attention Network**: Graph Attention Network (GAT) represents another cutting-edge model for graph data structures widely applied in real-world scenarios. 
3. **Graph Attention Network V2**: The second version introduces dynamic attention, enhancing the model's capabilities compared to the static attention in the first version.

See the Google Colab Notebooks stored in notebook folder for implementation details. 

## References
Brody, S. & Alon, U. & Yahav, E. 2022. How Attentive Are Graph Attention Networks? Arxiv. https://arxiv.org/pdf/2105.14491v3.pdf

Chen, X. et al. 2023. Activating More Pixels in Image Super-Resolution Transformer. Arxiv. https://arxiv.org/pdf/2205.04437.pdf

Dosovitskiy, A. et al. 2020. An Image is Worth 16 × 16 Words: Transformer for Image Recognition at Scale. Arxiv. https://arxiv.org/abs/2010.11929

Kipf, N. T. & Welling, M. 2017. Semi-Supervised Classification with Graph Convolutional Networks. Arxiv. https://arxiv.org/pdf/1609.02907.pdf

Liu, Z. et al. 2021. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. Arxiv. https://arxiv.org/abs/2103.14030

Quinn, M. & Sivesind, G. & Reis, G. n.d. Real-time Emotion Recognition From Facial Expressions. Stanford University. http://cs229.stanford.edu/proj2017/final-reports/5243420.pdf

Velickovic, A. et al. 2018. Graph Attention Networks. Arxiv. https://arxiv.org/pdf/1710.10903.pdf

Verma, R. 2018. Fer2013. Kaggle. https://www.kaggle.com/datasets/deadskull7/fer2013

Wang, Q. et al. 2021. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. Arxiv. https://arxiv.org/pdf/1910.03151.pdf

Woo, S. et al. 2018. CBAM: Convolutional Block Attention Module. Arxiv. https://arxiv.org/pdf/1807.06521.pdf
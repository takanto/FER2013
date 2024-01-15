# FER2013
This project is designed to provide developers with a range of pretrained models trained on FER2013. The project includes a Google Colab Notebook and pretrained TFLite files. (For GNNs, the approach involves converting PyTorch models to ONNX format and subsequently to TFLite files.)

## Dataset
FER2013 comprises 35,887 RGB images of faces, each with dimensions (48, 48, 3). These images represent 7 distinct emotions: happy, sad, angry, surprised, disgust, fear, and neutral (Verma, R., 2018). The dataset was curated by conducting a Google image search for each emotion and its synonyms (Quinn, M., Sivesind, G., Reis, G., n.d.).

## Data Preprocessing
The images undergo augmentation through the addition of noise, blurring, contrast changes, flipping, and warping. This process aims to expand the dataset and enhance model robustness. In the case of XGBoost and VGG16, dimensionality reduction is executed on the images leveraging a CNN Autoencoder. As for the Vision Transformer model, pre-trained on 224 by 224 images, upscaling is applied using the nearest neighbor algorithm. In the context of GNNs, facial landmarks are extracted to construct graphs.

## Models
The following is the list of models available in this repository. 

1. **XGBoost**: Renowned for its superior performance, XGBoost stands out due to its optimized regularizations and computations.
2. **VGG16**: A highly effective pretrained model for image classification tasks, VGG16's last layer is unfrozen and connected to a small ANN classifier with a softmax layer at the end.
3. **Vision Transformer**: Demonstrating impressive performance across image classification tasks, the Vision Transformer (ViT) segments 224 by 224 images into 16 by 16 patches of 4 by 4 pixels. These patches, along with positional embeddings, are then processed through a transformer. 
4. **Graph Convolutional Network**: Known as a state-of-the-art model for graph data structures, Graph Convolutional Network (GCN) excels in handling graph-based data.
5. **Graph Attention Network (v2)**: Graph Attention Network (GAT) represents another cutting-edge model for graph data structures widely applied in real-world scenarios. The second version introduces dynamic attention, enhancing the model's capabilities compared to the static attention in the first version.

## References
Quinn, M. & Sivesind, G. & Reis, G. n.d. Real-time Emotion Recognition From Facial Expressions. Stanford University. http://cs229.stanford.edu/proj2017/final-reports/5243420.pdf

Verma, R. 2018. Fer2013. Kaggle. https://www.kaggle.com/datasets/deadskull7/fer2013

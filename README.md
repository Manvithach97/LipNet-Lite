# LipReadingNeuralNetwork
LipNet paper implementation using Tensorflow

This repository contains the code and resources for reproducing the LipNet model based on the LipNet paper [here](https://arxiv.org/pdf/1611.01599). The model is built using TensorFlow and focuses on lip reading at the character level.

## Model Overview

The LipNet model operates at the character level and uses the following components:

- **Spatiotemporal Convolutional Neural Networks (STCNNs)**
- **Recurrent Neural Networks (RNNs)**
- **Connectionist Temporal Classification (CTC) Loss**

### Key Features

- **CTC Loss Function**: The Connectionist Temporal Classification (CTC) loss calculates the loss between a continuous (unsegmented) time series and a target sequence. Unlike the standard cross-entropy function, CTC avoids repetition of letters.
- **Bidirectional LSTM**: Used to account for temporal changes in letter recognition, allowing the model to process the sequence both forwards and backwards.

## Dataset

The dataset used is the **Grid Corpus (Pre-Processed)**. It is pre-processed specifically for training LipNet models. The preprocessing steps include:

1. Scaling down the video frame to focus on the mouth area.
2. Converting the videos to greyscale.
3. Converting the greyscaled videos to tensors.
4. Converting the tensors to GIFs for training.

## Streamlit App

A Streamlit app is included to provide a user-friendly interface for demonstrating how the model works. All the necessary files for running the app are provided in this repository.



## Running the Model

To run and test the model, follow these steps:

1. **Import Libraries**: Ensure all necessary libraries are imported.
2. **Download Dataset**: Run the first `gdown` cell to download and unzip the dataset.
3. **Run Model Compilation**: Execute all the cells up to the model compilation step.
4. **Use Pre-trained Weights**: No need to train the model again. I have provided the trained weights (trained till the 96th epoch) via Google Drive. Run the next `gdown` cell to import and use these weights.
5. **Test the Model**: Test the model on videos from the dataset by running the subsequent cells.

We have trained the whole dataset till 50 epochs
For each each we have saved the weights using checkpoints.


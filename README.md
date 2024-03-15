# Image-classification-via-brain-like-learning-in-neural-networks

## Background

### About SNNTorch
SNNTorch is an open-source library based on PyTorch for building and training spiking neural networks (SNNs). SNNs are a biologically inspired neural network model that differs from traditional artificial neural networks (ANNs) in that they use spike encoding and time encoding to represent and process information. SNNTorch provides a range of tools and functionalities to help users build, train, and evaluate SNN models, especially in event-driven environments.

### About MNIST
MNIST is a handwritten digit recognition dataset containing a series of handwritten digit images (0 to 9). This dataset is commonly used to test and validate the performance of machine learning algorithms, especially in image classification tasks. Each image is a grayscale image with a size of 28x28 pixels.

### About NMNIST
NMNIST is a spike-based handwritten digit recognition dataset. Unlike the traditional MNIST dataset, NMNIST uses spike encoding to represent image information. Spike encoding is a biologically inspired encoding method that mimics the way neurons process information. The NMNIST dataset is typically used to test and validate the performance of spiking neural networks (SNNs) and other neural network models.

### About DVSGesture
DVSGesture is a gesture recognition dataset containing gesture actions captured using a Dynamic Vision Sensor (DVS) camera. Unlike traditional cameras, the DVS camera outputs image data in the form of asynchronous events, which is closer to the working mechanism of the biological visual system. The DVSGesture dataset is commonly used to test and validate the performance of dynamic gesture recognition algorithms.

## Project Files

In this project, I compare the differences between SNN and ANN in terms of accuracy and loss and give my own explaination to the facts that I observed. 

In ANN.ipynb, I construct a CNN using pytorch and training with MNIST. After Training I use GradCam to visualize the weight distribution of the network in classfying a digit.

In SCNN(beta=.5).ipynb, SCNN(beta=.8).ipynb, SCNN(beta=.8, pop_coding).ipynb, I construct SNNs using SNNTorch with NMNIST. By choosing different beta values and output layers, I compare their performance and visualize the weight distribution. 

In SCNN(beta=.8)_DVSGesture.ipynb and SLSTM.ipynb, I compare the performance of Spiking CNN and Spiking LSTM in classfiying gestures. 

Models are saved as a file with the extension .pth. spk_plot.mp4 is presents how the SNN generates the result and accumulate the output of each time step. output_video.mp4 is the weight distribution of SNN in classfying DVSGesture.

## Running the Code

This repository contains the file of the project. Before running the notebook, you need to add some code to grad-cam's package. 

Python\Python310\lib\site-packages\pytorch_grad_cam\base_cam.py

at line 85

```python
if targets is None:
    target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
   targets = [ClassifierOutputTarget(
        category) for category in target_categories]
```

add the code in forward() below 'if targets is None:'

```python
if len(outputs)>1:
    outputs=outputs[0]
```

Then it looks like:

```python
if targets is None:
    #SNN has 2 outputs, spk_out and mem_out. Here we take the first output. 
    if len(outputs)>1:
        outputs=outputs[0]
    target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
   targets = [ClassifierOutputTarget(
        category) for category in target_categories]
```
After that, you can run through every notebook files.

# Traffic-Sign-Detection-and-Recognition-Deep-Learning-Model
Deep Learning model for Self-Driving Cars to reduce accidents with high accuracy in classification of 99.2%  and detection of 93%.

Using in the development Keras, TensorFlow, python, jupyter notebooks, deep learning, and computer vision techniques.

Accuracy in the detection is 93% and in the classification is 99.2%.

System for assisting the drivers to avoid accidents are becoming more and more important as the number of vehicles on road is on an exponential increase.
Advanced driver assistant systems is being effectively used in automobiles for providing lane keep assistance, forward collision warning, pedestrian warning driver drowsiness
detection traffic sign assist system etc. These form essential systems in autonomous cars for contextual awareness and road attribute mapping
in order to control the vehicle motion trajectory. Traffic sign recognition is the core component of traffic assist system for 
providing timely instructions and warning to the driver regarding traffic restrictions and information. In self-driving cars, the inputs
from the system used to make suitable decisions by the car for examples, to reduce speed or prepare for a detour. 
Traffic sign recognition involves traffic sign detection and classification.

First part is approach for traffic sign recognition based on YOLOv3 is presented.
The traffic sign recognition pipeline, consists of YOLOv3 detector trained for detecting the candidate traffic signs, a bounding box pre-processor,
which enlarges the detected bounding box, crops and resizes the boxes containing candidate traffic signs, and a CNN based classifier which
classifies the candidate traffic sign as belonging to one of the 43 classes.

Second part is The classifier network structure is in Table I. In this architecture, an n ×
n convolution is replaced by an n × 1 convolution followed by a 1 × n convolution, which reduces both the number of convolution operations
and the network parameters. This leads to computational cost reduction and increased speed. 
Batch Normalization and ReLU layers follows all layers other than the final dense layer. 
The sixth layer forms an inception module where kernels of different sizes are used to extract information from feature map output of the previous layer. 
The output feature maps from the inception layer are concatenated to combine the feature maps. Dropout layers are also used to regularize the activations of the final stages. 
To recognize the 43 traffic sign classes, fully-connected layer of an output size of 43 with Softmax activation is used as the last layer.

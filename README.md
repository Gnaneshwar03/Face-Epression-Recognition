# Facial Expression Recognition

This repository contains code for real-time face expression recognition using the EfficientNetB2 model. The project captures video frames from a webcam feed, detects faces using the dlib library, and performs emotion recognition on each detected face.

## Sample Image

https://github.com/Gnaneshwar03/Facial-Expression-Recognition/assets/107243397/b387b218-c9c2-4206-9bc2-86efb8a12859
<img src="https://github.com/Gnaneshwar03/Facial-Expression-Recognition/assets/107243397/b387b218-c9c2-4206-9bc2-86efb8a12859" width="700" height="400">


## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- TensorFlow
- numpy
- OpenCV (cv2)
- dlib
- pickle

You can install these dependencies using pip:

```
pip install tensorflow numpy opencv-python dlib
```

## Usage

To use the face expression recognition system:

1. Clone this repository to your local machine.

```shell
git clone https://github.com/Gnaneshwar03/Face-Expression-Recognition.git
```

2. Change to the project directory.

```shell
cd Face-Expression-Recognition
```

3. Download the necessary model weights and label encoder.

   - Download the pre-trained model weights file "best_weights.h5" from https://drive.google.com/file/d/16ccx1CbmUer9DEcKptS-0OW3dg2QA8ax/view?usp=drive_link and place it in the project directory.
   - Download the LabelEncoder file "LabelEncoder.pck" and place it in the project directory.

4. Run the real-time face expression recognition code.

```shell
python Face_Expression_Recognition_Realtime.py
```

5. The code will open a window displaying the webcam feed. It will detect faces in real-time and overlay the predicted emotion label on each detected face.

6. Press the 'q' key to quit the program.

## Model Architecture

The face expression recognition model architecture is based on EfficientNetB2. It consists of the EfficientNetB2 backbone followed by global average pooling, dropout, and dense layers. The output layer uses softmax activation to predict the emotion label.

## Files

- `Face_Expression_Recognition_Realtime.py`: This file contains the code for real-time face expression recognition. It captures video frames from the webcam feed, detects faces using dlib, and performs emotion recognition on each detected face.

- `best_weights.h5`: This file contains the trained weights of the face expression recognition model. It is required for loading the model architecture and making predictions.

- `LabelEncoder.pck`: This file contains the serialized LabelEncoder object used for encoding and decoding the emotion labels. It is required for inverse transforming the predicted emotion labels.

## Face Detection using PCA with People and Expression Classifiers

### Real-time detection implementing:
  - Principal Component Analysis (PCA)
  - Nearest Neighbour Classifier (SSD)
  - Support Vector Machines (SVM)
  
### Video demos:
- Nearest Neighbour Classifier (SSD)
  - Extreme recognition of single person
    - https://youtu.be/bO-e7UspUXU 
  - Recognition of people within and outside the recognised list. Tested using people found on Google Images
    - https://youtu.be/FJaFxk28X5Y
- Support Vector Machine Classifer (SVM)
  - Demo oriented around extreme recognition of single person
    - https://youtu.be/Zb_AllFWQ64
  - Recognition of people within and outside the recognised list. Tested using people found on Google Images
    - https://youtu.be/3laCd__XjS4

### Scripts to run:
- trainFace.py
  - Run this script to:
    - Fit the PCA model
    - Project faces in training images to eigenspace
    - Train/test person and expression classifiers
- rtFaceID.py
  - Run this script to perform real-time face detection
- videoFaceID.py
  - Run this script to perform face detection on a video file

### Custom PCA Module
- rxPCA.py

### Data pre-processing scripts:
- adjustDict.py
  - Drop eig coordinates of people to be removed from training set (reduce number of person classes)
- adjustDict(Remove expressions).py
  - Drop eigen coordinates of certain expressions (remove invalid expression classes. e.g. denoting head rotation instead of actual expressions)
- centroidFinder.py
  - Calculate eigen coordinate centroid for updating old pcDict.bin files without centroid data
- faceCropper.py
  - Tool for cropping faces from images
- reSizer.py
  - Image resizer script
- rxPCAtest.py
  - Script for testing custom PCA module
- suffixPhotos.py
  - Script for adding suffix to photo names

### Expression Analysis script
- expressionsAnalysis.q
  - Perform exploratory data analysis on eigen coordinates of expressions
  - Expression distribution can be plotted using Kx Developer
  
Classifiers Tested during training step:
- Nearest Neighbour Classifier (SSD)
- Support Vector Machines
- Adaboost
- GradientBoost
- XGBoost
- Stack Generalizer
- Random Forest
- Histogram Gradient Boosting
- K-NN 
- Extreme Learning Machines
    
Images and video-stream are zero-meaned to deal with lighting changes.
  
SSD used to determine if person in frame is not within training set.
  
EE4208 Intelligent System Design Module Assignment 1 Group Coursework

Due to privacy concerns, the training dataset and generated models are not included in this repository.  
To fit the PCA principal components and classification models, place your photos in a folder named `input`, with the notation: `person_expression.jpg`  
E.g. `./input/john_smiling.jpg`
  

Team Members:
- Calista Lee
- Chew Simin
- Choo Yaw Feng
- Foo Ren Xiang
- Zhang Zeyu
  
### Libraries Used:
- Scikit-Learn  
- OpenCV-Python  
- Python-ELM by D. Lambert  
  - https://github.com/dclambert/Python-ELM

### Dataset Used:
- CAS-PEAL Face Database (neutral and expressions sets)
  - http://www.jdl.ac.cn/peal/
- Group Member Photos
  - 5 photos of each person with varying head tilt  
  
429 images of 83 people with varying expressions were used to train expression classifiers  
112 images of 22 people were used to train person classifiers  

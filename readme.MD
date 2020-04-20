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
